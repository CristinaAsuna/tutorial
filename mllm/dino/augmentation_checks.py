"""Reject content-free crops and fixed-position masks, not just wrong shapes."""
import torch


def check_multicrop(make_views,make_mask):
    image=torch.linspace(0,1,40*40).reshape(1,1,40,40).expand(8,3,-1,-1).clone()
    torch.manual_seed(19);a=make_views(image)
    torch.manual_seed(19);repeat=make_views(image)
    torch.manual_seed(19);shifted=make_views(image+2)
    torch.manual_seed(23);other=make_views(image)
    assert len(a)==6 and all(v.shape==(8,3,32,32) for v in a[:2]) and all(v.shape==(8,3,16,16) for v in a[2:])
    for x,y,z in zip(a,repeat,shifted):
        torch.testing.assert_close(x,y,rtol=0,atol=0)
        torch.testing.assert_close(z,x+2,rtol=1e-5,atol=1e-5,msg='crop must preserve input content')
        assert x.var()>0 and float(x.min())>=0 and float(x.max())<=1
    assert any(not torch.equal(x,y) for x,y in zip(a,other)), 'crop must change with random seed'
    assert any(not torch.equal(a[0][0],a[0][i]) for i in range(1,8)), 'sample crop per image, not per batch'
    torch.manual_seed(19);mask=make_mask(8,64,.5,torch.device('cpu'))
    torch.manual_seed(19);repeat=make_mask(8,64,.5,torch.device('cpu'))
    torch.manual_seed(23);other=make_mask(8,64,.5,torch.device('cpu'))
    assert mask.dtype==torch.bool and mask.shape==(8,64) and (mask.sum(1)==32).all()
    assert torch.equal(mask,repeat), 'same seed must reproduce mask'
    assert not torch.equal(mask,other), 'fixed-position masks must fail'
    assert mask.unique(dim=0).shape[0]>1, 'rows need independent random masks'
