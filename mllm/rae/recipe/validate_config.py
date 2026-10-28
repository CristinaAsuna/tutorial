"""仅验证metadata与路径；不下载、不训练、不声称指标复现。"""
import argparse
import json
from pathlib import Path


def validate(config):
    if config['paper_version']!='2510.11690v1': raise ValueError('this course fixes paper v1')
    if not config['encoder_frozen'] or config['encoder_affine_layernorm']: raise ValueError('frozen non-affine encoder required')
    if config['decoder_noise_distribution']!='half_normal': raise ValueError('paper-v1 amplitude must be half-normal; main uniform differs')
    if config['time_direction']!='data_0_noise_1' or config['reference_dim']!=4096: raise ValueError('paper-v1 time/shift mismatch')
    if config['decoder_objectives']!=['L1','LPIPS','GAN']: raise ValueError('real decoder objective metadata incomplete')
    if config['training_executed'] or config['paper_metrics_reproduced']: raise ValueError('metadata skeleton has no training/metrics evidence')
    return {'metadata_valid':True,'data_exists':Path(config['data_path']).exists(),'weights_exist':Path(config['weights_path']).exists(),'training_executed':False,'paper_metrics_reproduced':False}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    args=parser.parse_args();print(json.dumps(validate(json.loads(args.config.read_text())),indent=2))

if __name__=='__main__':main()
