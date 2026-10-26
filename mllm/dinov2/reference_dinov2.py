"""DINOv2 algorithm course, single-process CPU. Not the released architecture/scale.

Shared family primitives live in the original DINO course; generic ViT operations
live in edu_core. No lesson imports here. Losses consume probabilities, not labels.
"""
from __future__ import annotations
import copy
import math
from types import SimpleNamespace
import torch
from torch import nn
import torch.nn.functional as F
from edu_core.training import freeze_and_keep_eval, update_ema, cosine_ema_momentum
from mllm.dino.reference_dinov2 import MiniViT, DINOHead
from mllm.dino.multicrop import MultiCropAugmentation


def rectangle_mask(grid, box, device=None):
    """(H,W), (top,left,height,width) -> HxW bool; True means hidden."""
    h, w = grid
    top, left, bh, bw = box
    if min(h,w,bh,bw) <= 0 or min(top,left) < 0 or top+bh > h or left+bw > w:
        raise ValueError("invalid patch rectangle")
    mask = torch.zeros(h,w,dtype=torch.bool,device=device)
    mask[top:top+bh,left:left+bw] = True
    return mask


def mask_count(num_patches, selected, ratio):
    return max(1, round(num_patches*ratio)) if selected else 0


def sample_masks(batch_size, grid, device, probability=.5, ratio_range=(.1,.5), rectangle_fn=rectangle_mask, count_fn=mask_count):
    """Block unions with per-sample ratio. Bounded search + exact-count toy fallback.

    Unlike the large-scale collator, independently sample each row's Bernoulli and
    ratio. Fallback fills residual random locations; document this toy difference.
    """
    if batch_size <= 0 or not 0 <= probability <= 1 or not 0 < ratio_range[0] <= ratio_range[1] <= 1:
        raise ValueError("invalid mask configuration")
    h,w = grid
    if min(h,w) <= 0:
        raise ValueError("positive grid required")
    masks = torch.zeros(batch_size,h,w,dtype=torch.bool,device=device)
    for row in masks:
        selected = float(torch.rand((),device=device)) < probability
        ratio = float(torch.empty((),device=device).uniform_(*ratio_range))
        count = count_fn(h*w, selected, ratio)
        if not count:
            continue
        for _ in range(20):
            left_count = count-int(row.sum())
            if left_count <= 0:
                break
            area = int(torch.randint(1,left_count+1,(),device=device))
            bh = min(h,max(1,round(math.sqrt(area))))
            bw = min(w,max(1,area//bh))
            top = int(torch.randint(h-bh+1,(),device=device)); left = int(torch.randint(w-bw+1,(),device=device))
            block = rectangle_fn(grid,(top,left,bh,bw),device)
            if int((block & ~row).sum()) <= left_count:
                row |= block
        remaining = count-int(row.sum())
        if remaining:
            available = (~row).flatten().nonzero().flatten()
            chosen = available[torch.randperm(len(available),device=device)[:remaining]]
            row.flatten()[chosen] = True
    return masks.flatten(1)


def build_modules(dim=32, prototypes=16, shared_head=False):
    student = nn.ModuleDict({'backbone': MiniViT(embed_dim=dim,depth=1,num_heads=4),
                             'dino_head': DINOHead(dim,prototypes,64,16)})
    student['ibot_head'] = student['dino_head'] if shared_head else DINOHead(dim,prototypes,64,16)
    teacher = nn.ModuleDict({'backbone': copy.deepcopy(student['backbone']),
                            'dino_head': student['dino_head'].teacher_copy()})
    teacher['ibot_head'] = teacher['dino_head'] if shared_head else student['ibot_head'].teacher_copy()
    freeze_and_keep_eval(teacher)
    return student,teacher


@torch.no_grad()
def sinkhorn(logits, temperature=.07, iterations=3):
    """NxK logits -> detached NxK assignments. Float64 log-space for stability.

    Each iteration: prototype row normalization, then sample column normalization.
    Columns of Q sum to 1/N exactly; prototype marginal approaches 1/K.
    """
    if logits.ndim != 2 or min(logits.shape) <= 0 or temperature <= 0 or iterations <= 0:
        raise ValueError("nonempty NxK, positive temperature and iterations required")
    if not torch.isfinite(logits).all():
        raise ValueError("finite logits required")
    logq = (logits.detach().double()/temperature).T
    k,n = logq.shape
    logq -= torch.logsumexp(logq.flatten(),0)
    for _ in range(iterations):
        logq -= torch.logsumexp(logq,dim=1,keepdim=True)+math.log(k)
        logq -= torch.logsumexp(logq,dim=0,keepdim=True)+math.log(n)
    return (logq+math.log(n)).exp().T.to(logits.dtype)


def dino_loss(students, targets, student_temp=.1):
    if len(targets) != 2 or len(students) < 2 or student_temp <= 0:
        raise ValueError("two teacher globals and at least two student views required")
    terms = [-(target.detach()*F.log_softmax(s/student_temp,-1)).sum(-1).mean()
             for i,target in enumerate(targets) for j,s in enumerate(students) if i != j]
    return torch.stack(terms).mean()


def ibot_loss(student, targets, patch_mask, student_temp=.1):
    """Per-image average over masked patches, then average over ALL B rows."""
    if student.shape != targets.shape or patch_mask.shape != student.shape[:2] or patch_mask.dtype != torch.bool or student_temp <= 0:
        raise ValueError("matching BNK logits/targets and BN bool patch_mask required")
    ce = -(targets.detach()*F.log_softmax(student/student_temp,-1)).sum(-1)
    per_image = (ce*patch_mask).sum(-1)/patch_mask.sum(-1).clamp_min(1)
    return per_image.mean()


def koleo(features, eps=1e-8):
    """Nearest-other-sample distance on normalized CLS. Batch must have >=2."""
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("KoLeo needs B>=2")
    x = F.normalize(features.float(),dim=-1)
    with torch.no_grad():
        similarity = x@x.T
        similarity.fill_diagonal_(-torch.inf)
        nearest = similarity.argmax(-1)
    distances = (x-x[nearest]).norm(dim=-1).clamp_min(eps)
    return -distances.log().mean()


def teacher_temperature(step, warmup_steps=4, start=.04, end=.07):
    if step < 0 or warmup_steps < 0 or min(start,end) <= 0:
        raise ValueError("invalid temperature schedule")
    return end if not warmup_steps else start+(end-start)*min(step/warmup_steps,1.)


def ema_momentum(step, total_steps, start=.994):
    return cosine_ema_momentum(step,total_steps,start=start)


def training_step(model, crops, masks, optimizer, step, total_steps):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = model.losses(crops,masks,step)
    losses['loss'].backward()
    optimizer.step()
    model.update_teacher(model.ops.ema_momentum(step,total_steps))
    return {key:value.detach() for key,value in losses.items()}


def reference_ops():
    return SimpleNamespace(sample_masks=sample_masks, build_modules=build_modules, sinkhorn=sinkhorn,
                           dino_loss=dino_loss, ibot_loss=ibot_loss, koleo=koleo,
                           teacher_temperature=teacher_temperature, ema_momentum=ema_momentum,
                           training_step=training_step)


class DINOv2(nn.Module):
    def __init__(self, ops=None, dim=32, prototypes=16, shared_head=False, centering='sinkhorn', koleo_weight=.1):
        super().__init__()
        if centering not in ('sinkhorn','center'):
            raise ValueError("centering must be sinkhorn or center")
        self.ops = ops or reference_ops()
        self.student,self.teacher = self.ops.build_modules(dim,prototypes,shared_head)
        self.centering, self.koleo_weight = centering,koleo_weight
        self.register_buffer('cls_center',torch.zeros(1,prototypes))
        self.register_buffer('patch_center',torch.zeros(1,prototypes))

    def train(self, mode=True):
        super().train(mode)
        freeze_and_keep_eval(self.teacher)
        return self

    @torch.no_grad()
    def update_teacher(self, momentum):
        update_ema(self.teacher,self.student,momentum)

    def views(self, images):
        crops = MultiCropAugmentation()(images)
        masks = [self.ops.sample_masks(c.shape[0],(c.shape[-2]//8,c.shape[-1]//8),c.device) for c in crops[:2]]
        return crops,masks

    def encode(self, modules, crops, masks=None):
        result=[]
        for i,crop in enumerate(crops):
            mask = masks[i] if masks is not None and i < 2 else None
            cls,patch = modules['backbone'](crop,mask)
            result.append({'features':cls,'patch_features':patch,
                           'cls':modules['dino_head'](cls),'patch':modules['ibot_head'](patch)})
        return result

    @torch.no_grad()
    def targets(self, logits, temp, center):
        if self.centering == 'sinkhorn':
            return self.ops.sinkhorn(logits,temp)
        target = ((logits-center)/temp).softmax(-1)
        center.lerp_(logits.mean(0,keepdim=True),.1)
        return target

    def losses(self,crops,masks,step,return_outputs=False):
        if len(crops) < 2 or len(masks) != 2:
            raise ValueError("globals first, two global masks required")
        with torch.no_grad():
            teacher = self.encode(self.teacher,crops[:2])
            temp = self.ops.teacher_temperature(step)
            cls_targets = self.targets(torch.cat([v['cls'] for v in teacher]),temp,self.cls_center).chunk(2)
            patches = torch.cat([v['patch'] for v in teacher])
            mask = torch.cat(masks)
            patch_targets = torch.zeros_like(patches)
            if mask.any():
                patch_targets[mask] = self.targets(patches[mask],temp,self.patch_center)
        student = self.encode(self.student,crops,masks)
        dino = self.ops.dino_loss([v['cls'] for v in student],cls_targets)
        ibot = self.ops.ibot_loss(torch.cat([v['patch'] for v in student[:2]]),patch_targets,mask)
        # Keep global views separate: no cross-view nearest-neighbor shortcut.
        reg = sum(self.ops.koleo(v['features']) for v in student[:2])
        losses = {'loss':dino+ibot+self.koleo_weight*reg,'dino':dino,'ibot':ibot,'koleo':reg}
        return (losses,student) if return_outputs else losses


def build_toy_dinov2(**kwargs):
    return DINOv2(**kwargs)
