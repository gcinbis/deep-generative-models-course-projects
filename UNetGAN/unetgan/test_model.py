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


def test_model():
    """
    Inference on multi GPU for distributed trained and saved model.
    Useful when whole model is saved instead of state dict.
    Probably useless otherwise.
    """
    make_deterministic()

    parser = ArgumentParser('Distributed Data Parallel')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local process rank.')
    args = parser.parse_args()
    
    args.is_master = args.local_rank == 0

    device = f'cuda:{args.local_rank}'

    distributed.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)

    map_location = {'cuda:%d' % 0: 'cuda:%d' % 0}
    model_obj = torch.load('outputs/2021_07_02--15.37.44-79.pth', map_location=map_location)
    # print(model_obj)
    # dis = model_obj['discriminator']
    gen = model_obj['generator']

    # print(gen)

    noise = torch.randn(
        size=(8, 128), device=device, requires_grad=False
    )

    g_z = gen(noise)

    print(g_z.shape)

    save_image(g_z, 'test_img.png', n_row=2)