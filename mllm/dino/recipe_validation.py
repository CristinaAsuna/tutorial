"""Recipe metadata checks, not an implementation of distributed training."""
from pathlib import Path


def validate_recipe(config,require_data=False):
    if config['paper'] not in ('dinov2','dinov3_gram'):
        raise ValueError('unknown course recipe')
    model,mask,opt=config['model'],config['masking'],config['optimization']
    if model['patch_size']<=0 or model['prototypes']<2 or not model['separate_heads']:
        raise ValueError('need positive patch size, >=2 prototypes and separate heads')
    if not 0<=mask['sample_probability']<=1 or not 0<mask['ratio_range'][0]<=mask['ratio_range'][1]<=1:
        raise ValueError('invalid masking')
    if min(opt['teacher_temp_start'],opt['teacher_temp_end'])<=0 or not 0<=opt['teacher_momentum_start']<=1:
        raise ValueError('invalid temperature/momentum')
    if config['targets']!='sinkhorn_knopp' or config['evaluation']!=['knn','linear_probe','dense_probe']:
        raise ValueError('record target normalization and all evaluation tracks')
    if config['paper']=='dinov3_gram':
        gram=config['gram']
        if not 0<gram['snapshot_step']<=gram['refine_start'] or gram['refresh_every']<=0 or gram['resolution_factor']!=2:
            raise ValueError('invalid Gram phase or resolution alignment')
        if gram['reduction'] not in ('mean','sum'):
            raise ValueError('explicit Gram reduction required')
    exists=Path(config['data_root']).is_dir()
    if require_data and not exists:raise FileNotFoundError(config['data_root'])
    return {'configuration_valid':True,'data_exists':exists,'training_executed':False,'paper_metrics_reproduced':False}
