import os
import argparse
import numpy as np
import time
from PIL import Image
from pathlib import Path
import cv2

import torch
import torch.nn.functional as F
import torchvision.transforms as TF
import torchvision.transforms.functional as FT

from lib.base.base_dataset import BaseDataset
import lib.models as models
from lib.utils import load_checkpoint, cam
from lib.eval.kid import calculate_kid_given_paths


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str,
                    default='./saved/UGATIT_selfie2anime/07-04_17-06/checkpoints/checkpoint-epoch4.pth',
                    help='Path to checkpoint')
parser.add_argument('--realA', type=str,
                    default='./data/selfie2anime/testA',
                    help='Real domainA images dir')
parser.add_argument('--realB', type=str,
                    default='./data/selfie2anime/testB',
                    help='Real domainB images dir')
parser.add_argument('--fakeB', type=str,
                    default='./saved/output_imgs/fakeB',
                    help='Generated domainB images dir')
parser.add_argument('--cam_dir', type=str,
                    default='./saved/output_imgs/cams',
                    help='save CAM outputs to dir')


def test(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model and load model checkpoint
    checkpoint = load_checkpoint(opt.checkpoint_path)
    config = checkpoint['config']
    ugatit = getattr(models, config.arch.type)(**config.arch.args)
    ugatit.to(device)
    ugatit.load_pretrained_weights(checkpoint['state_dict'])
    ugatit.eval()

    to_tensor = TF.Compose([
        TF.Resize((256, 256)),
        TF.ToTensor(),
        TF.Normalize(BaseDataset.MEAN, BaseDataset.STD)]
    )

    cam_out_dir = Path(opt.cam_dir)
    fakeB_out_dir = Path(opt.fakeB)
    cam_out_dir.mkdir(parents=True, exist_ok=True)
    fakeB_out_dir.mkdir(parents=True, exist_ok=True)
    images = Path(opt.realA).glob('*.jpg')
    for img_path in images:
        inp = Image.open(img_path).convert('RGB')
        inp = to_tensor(inp).unsqueeze(0).to(device)
        out, heatmap = ugatit.G_AB(inp)
        out = (out + 1.) / 2.
        out = out.detach().cpu().squeeze(0)
        out = FT.to_pil_image(out)
        out.save((fakeB_out_dir / img_path.name).with_suffix('.png'))

        cam_img = cam(heatmap.detach().cpu().squeeze(0).numpy().transpose(1, 2, 0), 256)
        cam_img = Image.fromarray(cam_img.astype('uint8'))
        cam_img.save((cam_out_dir / img_path.name).with_suffix('.png'))


if __name__ == '__main__':
    opt = parser.parse_args()
    print(opt)
    test(opt)
    mean, std = calculate_kid_given_paths(opt.realB, opt.fakeB, batch_size=2, cuda=True, dims=64, model_type='inception')
    print(f'KID mean:{mean:.2f} std:{std:.4f}')
