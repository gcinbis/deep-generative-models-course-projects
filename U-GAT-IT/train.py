import os
import sys
import json
import random
import argparse
import datetime
import numpy as np
from pathlib import Path
from easydict import EasyDict as edict

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from lib.utils import Logger
import lib.data as datasets
import lib.trainer as trainer


def main(config, resume, save_dir, log_dir):
    # DATA LOADERS
    train_set = getattr(datasets, config.datamanager.type)(
        config.datamanager.root, config.datamanager.dataset_dir,
        width=config.datamanager.width,
        height=config.datamanager.height,
        mean=config.datamanager.norm_mean,
        std=config.datamanager.norm_std,
        mode="train")
    val_set = getattr(datasets, config.datamanager.type)(
        config.datamanager.root, config.datamanager.dataset_dir,
        width=config.datamanager.width,
        height=config.datamanager.height,
        mean=config.datamanager.norm_mean,
        std=config.datamanager.norm_std,
        mode="test")

    train_loader = DataLoader(dataset=train_set, num_workers=config.datamanager.workers,
                              batch_size=config.datamanager.batch_size_train, shuffle=True,
                              pin_memory=config.datamanager.pin_memory)
    val_loader = DataLoader(dataset=val_set, num_workers=config.datamanager.workers,
                            batch_size=config.datamanager.batch_size_test, shuffle=False,
                            pin_memory=config.datamanager.pin_memory)

    print(f'\n{train_loader.dataset}\n')
    print(f'\n{val_loader.dataset}\n')

    # TRAINING
    runner = getattr(trainer, config.trainer.type)(
        config=config,
        resume=resume,
        save_dir=save_dir,
        log_dir=log_dir,
        train_loader=train_loader,
        val_loader=val_loader)

    runner.train()


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='./configs/selfie2anime.json', type=str,
                        help='Path to the config file (default: PSPNet.json)')
    parser.add_argument('-r', '--resume',
                        default="",
                        type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    # Setup seeds
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    cudnn.benchmark = True

    config = json.load(open(args.config))
    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    config = edict(config)

    start_time = datetime.datetime.now().strftime('%m-%d_%H-%M')
    save_dir = Path('.').resolve() / config.trainer.save_dir / config.name / start_time
    log_dir = save_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Logger(log_dir / 'train.txt')

    main(config, args.resume, save_dir, log_dir)
