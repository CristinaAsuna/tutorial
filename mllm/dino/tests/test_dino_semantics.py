"""Regression tests for actual algorithm blind spots found in the audit."""
import pytest
import torch
from mllm.dino.augmentation_checks import check_multicrop
from mllm.dino.multicrop import crop_and_resize, sample_crop_boxes, MultiCropAugmentation, make_patch_masks
from mllm.dino.reference_dinov2 import DINOiBOTLoss, MiniViT


def masks(b,n,r,device):
    return make_patch_masks([torch.zeros(b,3,8,8*n,device=device)],8,r)[0]


def test_real_augmentation_and_reject_content_free_views():
    check_multicrop(MultiCropAugmentation(),masks)
    def zeros(x):return [torch.zeros(x.shape[0],3,32,32)]*2+[torch.zeros(x.shape[0],3,16,16)]*4
    with pytest.raises(AssertionError):check_multicrop(zeros,masks)
    def fixed(b,n,r,device):
        out=torch.zeros(b,n,dtype=torch.bool);out[:,:round(n*r)]=True;return out
    with pytest.raises(AssertionError):check_multicrop(MultiCropAugmentation(),fixed)


def test_crop_content_and_area_fraction():
    x=torch.arange(32.).reshape(2,1,4,4)
    boxes=torch.tensor([[0,1,2,2],[2,0,2,2]])
    actual=crop_and_resize(x,boxes,2)
    torch.testing.assert_close(actual,torch.stack([x[0,:,0:2,1:3],x[1,:,2:4,0:2]]))
    boxes=sample_crop_boxes(torch.zeros(16,3,100,100),(.25,.25),aspect=(1.,1.))
    assert (boxes[:,2:]==50).all(), 'area .25 means side .5'


def test_teacher_temperature_endpoints_and_negative_step():
    loss=DINOiBOTLoss(4,warmup_steps=4)
    assert [round(loss.current_teacher_temp(i),3) for i in (0,2,4,8)]==[.04,.055,.07,.07]
    assert DINOiBOTLoss(4,warmup_steps=0).current_teacher_temp(0)==.07
    with pytest.raises(ValueError):loss.current_teacher_temp(-1)


def test_masked_input_pixels_cannot_leak_to_teacher_student_matching():
    torch.manual_seed(7)
    model=MiniViT(embed_dim=16,depth=1)
    image=torch.rand(1,3,32,32);changed=image.clone();changed[:,:,:8,:8]+=50
    mask=torch.zeros(1,16,dtype=torch.bool);mask[:,0]=True
    a=model(image,mask);b=model(changed,mask)
    for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
    # Unmasked teacher still sees the changed pixels.
    assert not torch.allclose(model(image)[1],model(changed)[1])


def test_center_and_temperature_use_old_buffer_then_update_for_next_step():
    import torch.nn.functional as F
    loss=DINOiBOTLoss(2,student_temp=.5,teacher_temp=.25,warmup_steps=0,center_momentum=.5,ibot_weight=0)
    old_center=torch.tensor([[.3,-.4]])
    loss.center.copy_(old_center)
    teachers=[{'cls':torch.tensor([[.2,.7]],requires_grad=True),'patch':torch.ones(1,1,2)},
              {'cls':torch.tensor([[.9,-.1]],requires_grad=True),'patch':torch.ones(1,1,2)}]
    students=[{'cls':torch.tensor([[1.,-1.]],requires_grad=True),'patch':torch.ones(1,1,2)},
              {'cls':torch.tensor([[-.2,.8]],requires_grad=True),'patch':torch.ones(1,1,2)}]
    expected=sum(-(F.softmax((teachers[i]['cls'].detach()-old_center)/.25,-1)*
                   F.log_softmax(students[1-i]['cls']/.5,-1)).sum() for i in range(2))/2
    actual=loss(students,teachers,[torch.ones(1,1,dtype=torch.bool)]*2,0)['loss']
    torch.testing.assert_close(actual,expected)
    actual.backward()
    assert all(t['cls'].grad is None for t in teachers)
    expected_center=old_center*.5+torch.tensor([[.55,.3]])*.5
    torch.testing.assert_close(loss.center,expected_center)
