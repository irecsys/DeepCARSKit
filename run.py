
# @Author : Yong Zheng


import argparse
import time
import torch
import multiprocessing as mcpu
from deepcarskit.quick_start import run
from logging import getLogger



if __name__ == '__main__':
    print('GPU availability: ', torch.cuda.is_available())

    n_gpu = torch.cuda.device_count()
    print('Num of GPU: ', n_gpu)

    if n_gpu>0:
        print(torch.cuda.get_device_name(0))
        print('Current GPU index: ', torch.cuda.current_device())

    logger = getLogger()
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', type=str, default='config.yaml', help='config files')

    args, _ = parser.parse_known_args()

    config_list = args.config_files.strip().split(' ') if args.config_files else None
    run(config_file_list=config_list)
    t1 = time.time()
    total = t1 - t0
    logger.info('time cost: '+ f': {total}s')