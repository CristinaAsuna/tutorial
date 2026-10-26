"""Provided architecture; lesson 2 handles head selection and teacher initialization."""
import copy
from torch import nn
from mllm.dino.reference_dinov2 import MiniViT, DINOHead


def build_modules(choose_patch_head, initialize_teacher, dim=32, prototypes=16, shared_head=False):
    student = nn.ModuleDict({'backbone':MiniViT(embed_dim=dim,depth=1,num_heads=4),
                             'dino_head':DINOHead(dim,prototypes,64,16)})
    student['ibot_head'] = choose_patch_head(student['dino_head'],DINOHead(dim,prototypes,64,16),shared_head)
    teacher = nn.ModuleDict({'backbone':copy.deepcopy(student['backbone']),
                             'dino_head':DINOHead(dim,prototypes,64,16)})
    teacher['ibot_head'] = choose_patch_head(teacher['dino_head'],DINOHead(dim,prototypes,64,16),shared_head)
    initialize_teacher(student,teacher)
    return student,teacher
