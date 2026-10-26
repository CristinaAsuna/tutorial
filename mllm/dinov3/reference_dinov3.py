"""DINOv3 section 4: Gram anchoring on the DINOv2 teaching backbone.

This deliberately isolates dense-feature refinement; it is NOT the full DINOv3
architecture/training recipe. Steps below count completed optimizer updates.
"""
from __future__ import annotations
import copy
import torch
import torch.nn.functional as F
from edu_core.training import freeze_and_keep_eval
from mllm.dino.multicrop import crop_and_resize, sample_crop_boxes, random_resized_crop
from mllm.dinov2.reference_dinov2 import DINOv2, reference_ops as v2_ops


def gram_matrix(features):
    """B,N,D -> B,N,N; normalize D, never N. No detach on student."""
    if features.ndim != 3:
        raise ValueError('need B,N,D features')
    x = F.normalize(features.float(),dim=-1)
    return x@x.transpose(-1,-2)


def gram_loss(student, teacher, reduction='mean', remove_neg=False, gram_fn=gram_matrix):
    if student.shape[:2] != teacher.shape[:2] or reduction not in ('mean','sum'):
        raise ValueError('matching patch grids and mean/sum reduction required')
    gs,gt = gram_fn(student),gram_fn(teacher.detach())
    if remove_neg:
        gs,gt = gs.clamp_min(0),gt.clamp_min(0)
    difference = (gs-gt).square()
    return difference.mean() if reduction == 'mean' else difference.sum()


def gram_event(completed_steps, snapshot_step=2, refine_start=4, refresh_every=2):
    if not 0 < snapshot_step <= refine_start or refresh_every <= 0 or completed_steps < 0:
        raise ValueError('need 0<snapshot<=refine, positive refresh')
    if completed_steps == snapshot_step:
        return 'snapshot'
    if completed_steps > refine_start and (completed_steps-refine_start)%refresh_every == 0:
        return 'refresh'
    return None


@torch.no_grad()
def sync_gram_teacher(gram_teacher, ema_backbone):
    gram_teacher.load_state_dict(ema_backbone.state_dict())
    freeze_and_keep_eval(gram_teacher)


def aligned_crops(images, boxes, low_size=32, high_size=64):
    """Two resizes of EXACTLY the same pixel box, not two random crops."""
    return crop_and_resize(images,boxes,low_size),crop_and_resize(images,boxes,high_size)


def downsample_features(features, source_grid, target_grid):
    b,n,d = features.shape
    if source_grid[0]*source_grid[1] != n or min(*source_grid,*target_grid) <= 0:
        raise ValueError('grid must match row-major patch tokens')
    image = features.transpose(1,2).reshape(b,d,*source_grid)
    return F.interpolate(image,size=target_grid,mode='bicubic',align_corners=False,antialias=False).flatten(2).transpose(1,2)


def refinement_loss(base, student_features, teacher_features, gram_fn, weight=1.):
    anchor = sum(gram_fn(s,t) for s,t in zip(student_features,teacher_features,strict=True))/len(student_features)
    return base+weight*anchor,anchor


def training_step(model, crops, masks, high_crops, optimizer, step, total_steps):
    if step != int(model.completed_steps):
        raise ValueError('step must equal restored completed_steps')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = model.losses(crops,masks,step,high_crops=high_crops)
    losses['loss'].backward()
    optimizer.step()
    model.update_teacher(model.ops.ema_momentum(step,total_steps))
    model.after_step(step+1)
    return {key:value.detach() for key,value in losses.items()}


def reference_ops():
    ops = v2_ops()
    for name in ('gram_matrix','gram_loss','gram_event','sync_gram_teacher','aligned_crops',
                 'downsample_features','refinement_loss','training_step'):
        setattr(ops,name,globals()[name])
    return ops


class DINOv3(DINOv2):
    def __init__(self, ops=None, snapshot_step=2, refine_start=4, refresh_every=2, gram_weight=1., **kwargs):
        ops = ops or reference_ops()
        super().__init__(ops=ops,**kwargs)
        if not 0 < snapshot_step <= refine_start or refresh_every <= 0 or gram_weight < 0:
            raise ValueError('invalid Gram schedule')
        self.gram_teacher = copy.deepcopy(self.teacher['backbone'])
        freeze_and_keep_eval(self.gram_teacher)
        self.snapshot_step,self.refine_start,self.refresh_every = snapshot_step,refine_start,refresh_every
        self.gram_weight = gram_weight
        self.register_buffer('completed_steps',torch.tensor(0))
        self.register_buffer('gram_ready',torch.tensor(False))
        self.register_buffer('last_gram_update',torch.tensor(-1))

    def get_extra_state(self):
        return {'version':1,'snapshot_step':self.snapshot_step,'refine_start':self.refine_start,
                'refresh_every':self.refresh_every,'gram_weight':self.gram_weight}

    def set_extra_state(self,state):
        if state['version'] != 1:
            raise ValueError('unsupported Gram checkpoint version')
        for name in ('snapshot_step','refine_start','refresh_every','gram_weight'):
            setattr(self,name,state[name])

    def train(self,mode=True):
        super().train(mode)
        freeze_and_keep_eval(self.gram_teacher)
        return self

    def views(self,images):
        pairs = [self.ops.aligned_crops(images,sample_crop_boxes(images,(.5,1.))) for _ in range(2)]
        crops = [pair[0] for pair in pairs]+[random_resized_crop(images,16,(.2,.5)) for _ in range(4)]
        masks = [self.ops.sample_masks(images.shape[0],(4,4),images.device) for _ in range(2)]
        return crops,masks,[pair[1] for pair in pairs]

    @torch.no_grad()
    def after_step(self,completed_steps):
        event = self.ops.gram_event(completed_steps,self.snapshot_step,self.refine_start,self.refresh_every)
        if event is not None:
            self.ops.sync_gram_teacher(self.gram_teacher,self.teacher['backbone'])
            self.gram_ready.fill_(True)
            self.last_gram_update.fill_(completed_steps)
        self.completed_steps.fill_(completed_steps)
        return event

    def losses(self,crops,masks,step,high_crops=None):
        losses,student = super().losses(crops,masks,step,return_outputs=True)
        anchor = losses['loss'].new_zeros(())
        if step >= self.refine_start:
            if not bool(self.gram_ready) or high_crops is None or len(high_crops) != 2:
                raise ValueError('refinement requires early Gram snapshot and two aligned high crops')
            with torch.no_grad():
                targets=[]
                for high,low in zip(high_crops,crops[:2]):
                    _,features = self.gram_teacher(high)
                    source = (high.shape[-2]//8,high.shape[-1]//8)
                    target = (low.shape[-2]//8,low.shape[-1]//8)
                    targets.append(self.ops.downsample_features(features,source,target))
            losses['loss'],anchor = self.ops.refinement_loss(losses['loss'],[s['patch_features'] for s in student[:2]],
                                                             targets,self.ops.gram_loss,self.gram_weight)
        losses['gram'] = anchor
        return losses


def build_toy_dinov3(**kwargs):
    return DINOv3(**kwargs)
