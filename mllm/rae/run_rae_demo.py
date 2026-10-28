"""CPU两阶段闭环；默认reference，practice全部通过学生ops，不下载。"""
import argparse
import torch
import json
from pathlib import Path
from .scaffolding import toy_images


def run(implementation='reference', steps=5, model=None, output_dir=None):
    if steps < 3:
        raise ValueError('at least 3 updates required to observe zero-initialized upstream training')
    torch.manual_seed(23)
    torch.set_num_threads(1)
    if model is None:
        if implementation=='reference':
            from .reference_rae import build_toy_rae
        else:
            from .practice_rae import build_toy_rae
        model=build_toy_rae()
    images=toy_images(); model.set_stage(1)
    encoder={k:v.clone() for k,v in model.rae.encoder.state_dict().items()}
    initial_decoder=model.rae.decoder_out.weight.detach().clone()
    optimizer=torch.optim.Adam(model.rae.parameters(),lr=.002)
    stage1_losses=[]
    for _ in range(steps):
        loss1=model.ops.stage1_step(model,images,optimizer); stage1_losses.append(loss1.item())
    model.rae.eval()
    with torch.no_grad(): reconstruction=model.rae(images)
    assert not torch.equal(initial_decoder,model.rae.decoder_out.weight)
    assert all(torch.equal(v,model.rae.encoder.state_dict()[k]) for k,v in encoder.items())
    model.set_stage(2)
    frozen={k:v.clone() for k,v in model.rae.state_dict().items()}
    original={k:v.clone() for k,v in model.generator.state_dict().items()}
    optimizer=torch.optim.Adam(model.generator.parameters(),lr=.002)
    stage2_losses=[]
    for _ in range(steps):
        loss2=model.ops.stage2_step(model,images,optimizer); stage2_losses.append(loss2.item())
    for key in ('base_embed.weight','noisy_embed.weight','condition_projector.weight','out.weight'):
        assert not torch.equal(original[key],model.generator.state_dict()[key]),key
    assert all(torch.equal(v,model.rae.state_dict()[k]) for k,v in frozen.items())
    noise=torch.randn(2,16,64)
    generated=model.ops.sample_images(model,noise,8)
    repeated=model.ops.sample_images(model,noise,8)
    torch.testing.assert_close(generated,repeated)
    assert generated.shape==images.shape and torch.isfinite(generated).all()
    assert torch.isfinite(loss1) and torch.isfinite(loss2)
    assert not model.rae.training and not model.ema.training
    print(f'RAE CPU {implementation}: stage1 L1={loss1.item():.4f}, stage2 flow={loss2.item():.4f}; frozen/updates/EMA/shifted Euler passed')
    if output_dir is not None:
        destination=Path(output_dir);destination.mkdir(parents=True,exist_ok=True)
        (destination/'losses.json').write_text(json.dumps({'implementation':implementation,'stage1_l1':stage1_losses,'stage2_flow':stage2_losses,'paper_metrics_reproduced':False},indent=2))
        for name,image in [('input',images[0]),('reconstruction',reconstruction[0]),('generated',generated[0])]:
            # Map fixed display range [-1,1] to bytes; clipping is display-only.
            pixels=((image.detach().cpu().clamp(-1,1)+1)*127.5).round().to(torch.uint8)
            payload=bytes(pixels.permute(1,2,0).contiguous().flatten().tolist())
            (destination/(name+'.ppm')).write_bytes(b'P6\n16 16\n255\n'+payload)
    return model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--implementation',choices=('reference','practice'),default='reference')
    parser.add_argument('--steps',type=int,default=5)
    parser.add_argument('--output-dir')
    args=parser.parse_args()
    try: run(args.implementation,args.steps,output_dir=args.output_dir)
    except NotImplementedError as exc:
        print(str(exc)); return 2
    return 0

if __name__=='__main__':
    raise SystemExit(main())
