"""Validate REPA recipe metadata without downloading data or claiming training."""
import argparse
import json
from pathlib import Path


def validate_config(config, require_data=False):
    if config.get('time_direction')!='0=data,1=noise' or config.get('prediction')!='noise-minus-clean':
        raise ValueError('REPA code uses noise-minus-clean and decreasing sampling time')
    if config.get('alignment_objective')!='negative_patch_cosine' or config.get('alignment_weight',-1)<0:
        raise ValueError('negative patch cosine and nonnegative weight required')
    if not isinstance(config.get('alignment_depth'),int) or config['alignment_depth']<1:
        raise ValueError('alignment depth must be positive, one-based')
    if config.get('training_executed') is not False or config.get('paper_metrics_reproduced') is not False:
        raise ValueError('this metadata validator cannot attest real training or paper metrics')
    exists=Path(config['data_path']).exists()
    if require_data and not exists: raise FileNotFoundError(config['data_path'])
    return {'metadata_valid':True,'data_exists':exists,'training_executed':False,'paper_metrics_reproduced':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    p.add_argument('--require-data',action='store_true')
    a=p.parse_args()
    print(json.dumps(validate_config(json.loads(a.config.read_text()),a.require_data)))


if __name__=='__main__': main()
