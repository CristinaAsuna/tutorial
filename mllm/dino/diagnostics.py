"""Deterministic structured images and dependency-free SVG feature diagnostics."""
from pathlib import Path
import json
import torch
import torch.nn.functional as F


def patterned_images(batch_size=4,size=80):
    y,x = torch.meshgrid(torch.linspace(-1,1,size),torch.linspace(-1,1,size),indexing='ij')
    images=[]
    for i in range(batch_size):
        cx,cy = (-.3 if i%2 else .3),(-.2 if i//2%2 else .2)
        shape = ((x-cx).square()+(y-cy).square() < .25).float()
        stripe = ((x+(i+1)*y).sin()+1)/2
        images.append(torch.stack((shape, stripe*.5, (1-shape)*(.2+i*.1))))
    return torch.stack(images)


@torch.no_grad()
def feature_diagnostics(model,images):
    was_training=model.training
    model.eval()
    cls,patch = model.student['backbone'](F.interpolate(images,(32,32),mode='bilinear',align_corners=False))
    probs=(model.student['dino_head'](cls)/.07).softmax(-1)
    normalized=F.normalize(patch,dim=-1)
    gram=normalized@normalized.transpose(-1,-2)
    metrics={'sample_entropy':float(-(probs*probs.clamp_min(1e-12).log()).sum(-1).mean()),
             'prototype_entropy':float(-(probs.mean(0)*probs.mean(0).clamp_min(1e-12).log()).sum()),
             'patch_spatial_variance':float(normalized.var(dim=1,unbiased=False).mean()),
             'cls_batch_variance':float(F.normalize(cls,dim=-1).var(dim=0,unbiased=False).mean())}
    model.train(was_training)
    return metrics,gram[0].cpu()


def write_diagnostics(output_dir,records,matrices):
    """Standalone SVG heatmaps; labels explicitly describe synthetic diagnostics."""
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    (output/'metrics.json').write_text(json.dumps(records,indent=2,ensure_ascii=False)+'\n')
    size=190; width=len(matrices)*(size+30)+20
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="255" viewBox="0 0 {width} 255">',
         '<rect width="100%" height="100%" fill="white"/>',
         '<text x="15" y="20" font-size="13">Synthetic patch cosine maps (blue=-1, white=0, red=1)</text>']
    for panel,(label,matrix) in enumerate(matrices.items()):
        n=matrix.shape[0]; cell=size/n;left=15+panel*(size+30)
        svg.append(f'<text x="{left}" y="43" font-size="12">{label}</text>')
        for y in range(n):
            for x in range(n):
                v=float(matrix[y,x].clamp(-1,1)); fade=round(255*(1-abs(v)))
                color=f'rgb(255,{fade},{fade})' if v>=0 else f'rgb({fade},{fade},255)'
                svg.append(f'<rect x="{left+x*cell:.3f}" y="{52+y*cell:.3f}" width="{cell+.02:.3f}" height="{cell+.02:.3f}" fill="{color}"/>')
    svg.append('</svg>');(output/'patch_similarity.svg').write_text('\n'.join(svg))
