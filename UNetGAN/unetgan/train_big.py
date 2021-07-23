from argparse import ArgumentParser

import torch
import torchvision
import torchvision.transforms as transforms

import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torchvision.utils import save_image

from unetgan.utils import preview_samples
from .models.biggan import *

from .losses import adversarial_loss, pixelwise_adversarial_loss
from .data import CelebA
from .utils import preview_samples, get_cutmix_mask, make_deterministic, Logger
from .utils import freeze, unfreeze

import os
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt


def train_ubiggan(
    train_path, latent_dim, 
    batch_size=100, device='cuda', num_epochs=40, **kwargs
):
    """
    Train a 128x128 U-BigGAN network.
    """

    make_deterministic()

    # Distributed launch argument
    parser = ArgumentParser('Distributed Data Parallel')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local process rank.')
    args = parser.parse_args()
    
    args.is_master = args.local_rank == 0

    # Generate necessary output folders
    if args.is_master:
        run_name = datetime.now().strftime('%Y_%m_%d--%H.%M.%S')
        output_dir = os.path.join('outputs/runs', run_name)
        os.makedirs(output_dir, exist_ok=True)

        logger = Logger('outputs/logs', run_name)

    device = f'cuda:{args.local_rank}'

    # Distributed initialization
    distributed.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)

    dis = BigGANDiscriminator().to(device)
    gen = BigGANGenerator(latent_dim=latent_dim, base_ch_width=64).to(device)

    # Models as distributed data parallel
    dis = DDP(
        dis,
        device_ids=[args.local_rank],
        output_device=args.local_rank
    )
    gen = DDP(
        gen,
        device_ids=[args.local_rank],
        output_device=args.local_rank
    )

    # Hyperparameters
    image_size = kwargs.get('image_size', (128, 128))
    visualize_frequency = kwargs.get('visualize_frequency', 50)
    d_learning_rate = kwargs.get('d_learning_rate', 1e-3)
    g_learning_rate = kwargs.get('g_learning_rate', 2e-4)
    cutmix_warmup_iters = kwargs.get('cutmix_warmup_iters', 8000)

    # Torchvision supplied CelebA dataset
    train_set = CelebA(
        train_path, download=True,
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    )
    sampler = DistributedSampler(train_set)

    train_loader = torch.utils.data.DataLoader(
        train_set, sampler=sampler, batch_size=batch_size
    )
    train_loader_iter = iter(train_loader)

    # Optimizers for generator and discriminator
    optim_g = torch.optim.Adam(
        lr=g_learning_rate, params=gen.parameters(), betas=(0.5, 0.999)
    )
    optim_d = torch.optim.Adam(
        lr=d_learning_rate, params=dis.parameters(), betas=(0.5, 0.999)
    )

    # Iterate over batches, rather than epochs
    g_losses = []
    g_acc = 0
    d_losses = []
    d_acc = 0

    # Fix some noise to visualize every now and then
    fixed_noise = torch.randn(
        size=(8, latent_dim), device=device, requires_grad=False
    )
    
    if args.is_master:
        print(f"Epoch length {len(train_loader)}")
    iter_tqdm = tqdm(range(1, 1 + num_epochs * len(train_loader_iter)))
    for iteration in iter_tqdm:
        gen.train()
        dis.train()

        # DDP barrier for models to sync
        distributed.barrier()


        ### DISCRIMINATOR ###

        # Clear generator graph to free memory and freeze generator
        gen.zero_grad(set_to_none=True)
        freeze(gen)
        unfreeze(dis)

        # Update the discriminator twice in an iteration
        for _ in range(2):
            # Get the next batch
            try:
                image_batch, _ = train_loader_iter.next()
            except StopIteration:
                train_loader_iter = iter(train_loader)
                image_batch, _ = train_loader_iter.next()

            image_batch = image_batch.to(device)
            
            # Our batches may be "incomplete" if we're at the end
            input_batch_size = image_batch.shape[0]

            # Real / fake vectors for image discriminator
            real_labels = torch.ones(size=(input_batch_size, 1), device=device)
            fake_labels = torch.zeros(size=(input_batch_size, 1), device=device)

            # Real / fake image masks for per-pixel discriminator
            real_mask = torch.ones(size=(input_batch_size, 1, *image_size), device=device)
            fake_mask = torch.zeros(size=(input_batch_size, 1, *image_size), device=device)


            ## Calculate discriminator losses
            dis.zero_grad(set_to_none=True)
            
            # Create a batch of sample noise
            noise = torch.randn(size=(input_batch_size, latent_dim), device=device)
            g_z = gen(noise).detach()

            d_r_i = dis(image_batch)
            d_g_z = dis(g_z)

            # Cutmix sample probability
            cutmix_prob = torch.rand(1).item()
            # Start at 0, linearly scale to 0.5 after 1000 iterations
            cutmix_threshold = min(0.5, iteration * (0.5 / cutmix_warmup_iters))
            cutmix_examples = None
            masks = None
            compute_cutmix_loss = False
            if cutmix_prob < cutmix_threshold:
                compute_cutmix_loss = True
                # Maybe vectorize the mask fn if this is a bottleneck
                masks = []
                for _ in range(input_batch_size):
                    mask, _ = get_cutmix_mask(image_size)
                    masks.append(mask)
                masks = torch.stack(masks).unsqueeze(1).to(device)
                cutmix_examples = masks * image_batch + (1.0 - masks) * g_z.detach()

            # Get -Ex[logD(x)]
            d_real_enc_loss = adversarial_loss(d_r_i[0], real_labels)
            # Get -Ez[log(1-D(G(z)))]
            d_fake_enc_loss = adversarial_loss(d_g_z[0], fake_labels)
            ## Same for pixelwise discriminator loss
            # Get -Ex[logD(x)]
            d_real_dec_loss = pixelwise_adversarial_loss(d_r_i[1], real_mask)
            # Get -Ez[log(1-D(G(z)))]
            d_fake_dec_loss = pixelwise_adversarial_loss(d_g_z[1], fake_mask)

            # CutMix losses
            d_consistancy_loss = 0.0
            if compute_cutmix_loss:
                cutmix_out_enc, cutmix_out_dec = dis(cutmix_examples)
                d_cutmix_enc_loss = adversarial_loss(cutmix_out_enc, fake_labels)
                d_cutmix_dec_loss = pixelwise_adversarial_loss(cutmix_out_dec, masks)
                d_consistancy_loss = d_cutmix_enc_loss + d_cutmix_dec_loss

            d_enc_losses = d_real_enc_loss + d_fake_enc_loss
            d_dec_losses = d_real_dec_loss + d_fake_dec_loss

            # Don't need to get the mean here
            d_loss = d_enc_losses + d_dec_losses + d_consistancy_loss
            d_loss.backward()
            optim_d.step()
            d_losses.append(d_loss.item() / input_batch_size)
            d_acc += d_loss.item()


        ### GENERATOR ###

        # Clear discriminator graph to free memory and freeze discriminator
        dis.zero_grad(set_to_none=True)
        freeze(dis)
        unfreeze(gen)

        real_labels = torch.ones(size=(input_batch_size, 1), device=device)
        real_mask = torch.ones(size=(input_batch_size, 1, *image_size), device=device)

        ## Calculate non-saturating generator loss
        gen.zero_grad(set_to_none=True)

        # Create a batch of sample noise
        noise = torch.randn(size=(input_batch_size, latent_dim), device=device)
        g_z = gen(noise)

        # Get -Ez[D(G(z))]
        d_g_z = dis(g_z)
        g_loss_d_enc = adversarial_loss(d_g_z[0], real_labels)
        g_loss_d_dec = pixelwise_adversarial_loss(d_g_z[1], real_mask)

        g_loss = g_loss_d_enc + g_loss_d_dec

        g_loss.backward()
        optim_g.step()
        g_losses.append(g_loss.item() / input_batch_size)
        g_acc += g_loss.item()


        smooth_d_loss = d_acc / iteration
        smooth_g_loss = g_acc / iteration
        iter_tqdm.desc = f'D loss: {smooth_d_loss:.5f}, G loss: {smooth_g_loss:.5f}'
        if args.is_master:
            # Tensorboard logging
            logger.list_of_scalars_summary([
                ["Generator Loss", smooth_g_loss],
                ["Discriminator Loss", smooth_d_loss],
            ], iteration)


            if (iteration // len(train_loader)) % 10:
                model_obj = {
                    'discriminator': dis.state_dict(),
                    'generator': gen.state_dict(),
                }
                torch.save(model_obj, f'outputs/{run_name}-{iteration // len(train_loader)}.pth')
        
        # Reordering like this helps against the asymmetric update problem
        if iteration % visualize_frequency == visualize_frequency - 1:
            with torch.no_grad():
                gen.eval()
                dis.eval()
                samples = gen(fixed_noise)
                if args.is_master:
                    save_image((samples + 1) / 2, os.path.join(output_dir, f'{iteration+1:07d}.png'), n_row=2)

    return gen, dis, d_losses, g_losses

