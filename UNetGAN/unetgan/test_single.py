from argparse import ArgumentParser

import torch
import torchvision
import torchvision.transforms as transforms

import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from unetgan.utils import preview_samples
from .models.biggan import *

from .losses import adversarial_loss, pixelwise_adversarial_loss
from .data import CelebA
from .utils import preview_samples, get_cutmix_mask, make_deterministic

from tqdm import tqdm
import matplotlib.pyplot as plt


from torchvision.utils import save_image


def test_single():
    """
    Inference on single GPU for distributed trained and saved model.
    """

    make_deterministic(seed=218)

    device = f'cuda:0'

    gen = BigGANGenerator(latent_dim=140, base_ch_width=64)

    ckpt = torch.load('outputs/2021_07_02--15.37.44-79.pth')
    state_dict = ckpt['generator']

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v

    gen.load_state_dict(new_state_dict)

    gen = gen.to(device)

    noise = torch.randn(
        size=(8, 140), device=device, requires_grad=False
    )

    g_z = gen(noise)

    print(g_z.shape)

    save_image((g_z + 1) / 2, 'test_img1.png', n_row=2)
