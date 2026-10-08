"""只运行选中的关卡；reference 是数值 oracle，practice 默认且不回退。"""
import argparse
import math
import traceback
import torch
import torch.nn.functional as F
from reference_dinov2 import MiniViT, DINOiBOTLoss

def check(lesson, implementation):
    torch.manual_seed(7)
    reference = implementation == 'reference'
    if lesson == 1:
        if reference:
            cls = MiniViT
        else:
            from lesson1_vit import ExerciseViT as cls
        model = cls(patch_size=8, embed_dim=16, image_size=32, depth=1, num_heads=4)
        for size in (32,16):
            x = torch.randn(2,3,size,size)
            n = (size//8)**2
            mask = torch.zeros(2,n,dtype=torch.bool); mask[:,0]=True
            a,b = model(x, mask)
            assert a.shape == (2,16) and b.shape == (2,n,16), 'CLS/patch shape 或 local position 插值错误'
            (a.square().mean()+b.square().mean()).backward()
        assert model.mask_token.grad is not None, 'mask token 未参与计算'
    elif lesson == 2:
        from multicrop import MultiCropAugmentation, make_patch_masks
        if reference:
            views = MultiCropAugmentation()(torch.randn(2,3,40,40))
            mask = make_patch_masks([torch.zeros(2,3,16,16)],8,.5)[0]
        else:
            from lesson2_multicrop_masking import make_multicrop_views, random_patch_mask
            views = make_multicrop_views(torch.randn(2,3,40,40))
            mask = random_patch_mask(2,4,.5,torch.device('cpu'))
        assert len(views)==6 and all(v.shape==(2,3,32,32) for v in views[:2]) and all(v.shape==(2,3,16,16) for v in views[2:]), 'crop 顺序/尺寸错误'
        assert mask.dtype==torch.bool and mask.shape==(2,4) and (mask.sum(1)==2).all(), '每样本必须精确两 masked patches'
    elif lesson == 3:
        from lesson3_dino_loss import dino_cross_view_loss, update_center
        students = [torch.tensor([[1.,-1.]],requires_grad=True),torch.tensor([[0.,2.]],requires_grad=True),torch.tensor([[3.,0.]],requires_grad=True)]
        teachers = [torch.tensor([[2.,0.]],requires_grad=True),torch.tensor([[0.,2.]],requires_grad=True)]
        center=torch.zeros(1,2)
        expected = sum(-(F.softmax(t.detach(),-1)*F.log_softmax(s,-1)).sum(-1).mean() for i,t in enumerate(teachers) for j,s in enumerate(students) if i!=j)/4
        if reference:
            criterion = DINOiBOTLoss(2,student_temp=1.,teacher_temp=1.,warmup_steps=0,center_momentum=.5,ibot_weight=0.)
            outputs = [{'cls': s, 'patch': torch.zeros(1,1,2)} for s in students]
            target_outputs = [{'cls': v, 'patch': torch.zeros(1,1,2)} for v in teachers]
            result = criterion(outputs,target_outputs,[torch.ones(1,1,dtype=torch.bool)]*2,0)['loss']
        else:
            result = dino_cross_view_loss(students,teachers,center,1.,1.)
        torch.testing.assert_close(result,expected)
        result.backward(); assert all(t.grad is None for t in teachers), 'teacher 必须 detach'
        new = criterion.center if reference else update_center(center,teachers,.5)
        torch.testing.assert_close(new,torch.tensor([[.5,.5]])); assert not new.requires_grad, 'center 不能有计算图'
    else:
        from lesson4_ibot_teacher_student import ibot_masked_patch_loss, ema_update
        s=torch.tensor([[[0.,0.],[4.,-4.]]],requires_grad=True);t=torch.tensor([[[2.,0.],[0.,2.]]],requires_grad=True); mask=torch.tensor([[True,False]])
        expected=-(F.softmax(t.detach(),-1)*F.log_softmax(s,-1)).sum(-1)[mask].mean()
        if reference:
            criterion=DINOiBOTLoss(2,student_temp=1.,teacher_temp=1.,warmup_steps=0)
            outputs=[{'cls':torch.zeros(1,2),'patch':s}]*2
            targets=[{'cls':torch.zeros(1,2),'patch':t}]*2
            losses=criterion(outputs,targets,[mask,mask],0)
            result=losses['loss']-losses['dino_loss']
        else:
            result=ibot_masked_patch_loss(s,t,mask,torch.zeros(1,2),1.,1.)
        torch.testing.assert_close(result,expected); result.backward()
        assert torch.equal(s.grad[:,1],torch.zeros_like(s.grad[:,1])) and t.grad is None, 'unmasked 梯度必须为零，teacher 无梯度'
        teacher=torch.nn.Parameter(torch.tensor([2.]),requires_grad=False);student=torch.nn.Parameter(torch.tensor([6.]))
        if reference:
            with torch.no_grad():teacher.mul_(.75).add_(student,alpha=.25)
        else:ema_update([teacher],[student],.75)
        torch.testing.assert_close(teacher,torch.tensor([3.]));assert student.grad is None

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--lesson',type=int,choices=range(1,5));p.add_argument('--implementation',choices=['reference','practice'],default='practice');a=p.parse_args()
    try:
        for lesson in ([a.lesson] if a.lesson is not None else range(1,5)):
            check(lesson,a.implementation)
            print(f"lesson {lesson} {a.implementation} passed")
    except NotImplementedError as exc:
        traceback.print_exc();p.exit(2,f'未完成：第 {lesson} 关；按 traceback 文件/函数查找 TODO，参见 HINTS.md。\n')
    print(f'DINO lesson {lesson} {a.implementation} passed')
