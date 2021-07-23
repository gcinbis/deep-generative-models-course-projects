from unetgan.utils import preview_samples
from .models.dcgan import *

import torch
import torchvision.transforms as transforms

from .losses import adversarial_loss, pixelwise_adversarial_loss
from .data import CelebA
from .utils import preview_samples, get_cutmix_mask

from tqdm import tqdm
import matplotlib.pyplot as plt

def train(train_path, learning_rate, latent_dim, 
          batch_size=128, device='cpu', num_iters=10000, **kwargs):
    """
    Train a 64x64 DCGAN network.
    """

    dis = DCGANDiscriminator().to(device)
    gen = DCGANGenerator(nz=latent_dim).to(device)

    image_size = kwargs.get('image_size', (64, 64))
    loss_report_frequency = kwargs.get('loss_report_frequency', 100)

    train_set = CelebA(train_path, download=True, transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    train_loader_iter = iter(train_loader)

    optim_g = torch.optim.Adam(lr=learning_rate, params=gen.parameters(), betas=(0.5, 0.999))
    optim_d = torch.optim.Adam(lr=learning_rate, params=dis.parameters(), betas=(0.5, 0.999))

    # Iterate over batches, rather than epochs
    g_losses = []
    d_losses = []

    fixed_noise = torch.randn(size=(8, latent_dim), device=device, requires_grad=False)

    for iteration in tqdm(range(num_iters)):
        gen.train()
        dis.train()

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

        ## Calculate non-saturating generator loss
        gen.zero_grad()

        # Create a batch of sample noise
        noise = torch.randn(size=(input_batch_size, latent_dim), device=device)

        g_z = gen(noise)
        # Get -Ez[D(G(z))] 
        g_loss = adversarial_loss(dis(g_z), real_labels)

        g_loss.backward()
        optim_g.step()
        g_losses.append(g_loss.item() / input_batch_size)

        ## Calculate discriminator losses
        dis.zero_grad()
        # Get -Ex[logD(x)]
        d_real_loss = adversarial_loss(dis(image_batch), real_labels)
        # Get -Ez[log(1-D(G(z)))]
        d_fake_loss = adversarial_loss(dis(g_z.detach()), fake_labels)
        ## Same for pixelwise discriminator loss

        d_loss = (d_real_loss + d_fake_loss) / 2.0
        d_loss.backward()
        optim_d.step()
        d_losses.append(d_loss.item() / input_batch_size)

        if iteration % loss_report_frequency == loss_report_frequency - 1:
            print(f'Discriminator loss: {d_loss}, Generator loss: {g_loss}')
            # TODO: Validation here
            if iteration % (2 * loss_report_frequency)== 2 * loss_report_frequency - 1:
                # Visualize the fixed noise samples
                gen.eval()
                dis.eval()
                samples = gen(fixed_noise)
                preview_samples(plt, samples)
                plt.show()

    return gen, dis, d_losses, g_losses

def train_udcgan(train_path, latent_dim, 
          batch_size=128, device='cpu', num_iters=10000, **kwargs):
    """
    Train a 64x64 U-DCGAN network.
    """
    dis = UGANDiscriminator().to(device)
    gen = DCGANGenerator(nz=latent_dim).to(device)

    image_size = kwargs.get('image_size', (64, 64))
    loss_report_frequency = kwargs.get('loss_report_frequency', 100)
    d_learning_rate = kwargs.get('d_learning_rate', 3e-4)
    g_learning_rate = kwargs.get('g_learning_rate', 1e-4)

    train_set = CelebA(train_path, download=True, transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    train_loader_iter = iter(train_loader)

    optim_g = torch.optim.Adam(lr=g_learning_rate, params=gen.parameters(), betas=(0.5, 0.999))
    optim_d = torch.optim.Adam(lr=d_learning_rate, params=dis.parameters(), betas=(0.5, 0.999))

    # Iterate over batches, rather than epochs
    g_losses = []
    d_losses = []

    fixed_noise = torch.randn(size=(8, latent_dim), device=device, requires_grad=False)

    for iteration in tqdm(range(num_iters)):
        gen.train()
        dis.train()

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
        real_mask = torch.ones(size=(input_batch_size, 3, *image_size), device=device)
        fake_mask = torch.zeros(size=(input_batch_size, 3, *image_size), device=device)

        ## Calculate non-saturating generator loss
        gen.zero_grad()

        # Create a batch of sample noise
        noise = torch.randn(size=(input_batch_size, latent_dim), device=device)

        g_z = gen(noise)
        # Get -Ez[D(G(z))] 
        g_loss_d_enc = adversarial_loss(dis(g_z)[0], real_labels)
        g_loss_d_dec = pixelwise_adversarial_loss(dis(g_z)[1], real_mask)

        g_loss = g_loss_d_enc + g_loss_d_dec

        g_loss.backward()
        optim_g.step()
        g_losses.append(g_loss.item() / input_batch_size)

        ## Calculate discriminator losses
        dis.zero_grad()
        # Get -Ex[logD(x)]
        d_real_enc_loss = adversarial_loss(dis(image_batch)[0], real_labels)
        # Get -Ez[log(1-D(G(z)))]
        d_fake_enc_loss = adversarial_loss(dis(g_z.detach())[0], fake_labels)
        ## Same for pixelwise discriminator loss
        # Get -Ex[logD(x)]
        d_real_dec_loss = pixelwise_adversarial_loss(dis(image_batch)[1], real_mask)
        # Get -Ez[log(1-D(G(z)))]
        d_fake_dec_loss = pixelwise_adversarial_loss(dis(g_z.detach())[1], fake_mask)

        d_loss = (d_real_enc_loss + d_fake_enc_loss + d_real_dec_loss + d_fake_dec_loss) / 4.0
        d_loss.backward()
        optim_d.step()
        d_losses.append(d_loss.item() / input_batch_size)

        if iteration % loss_report_frequency == loss_report_frequency - 1:
            print(f'Discriminator loss: {d_loss}, Generator loss: {g_loss}')
            # TODO: Validation here
            if iteration % (2 * loss_report_frequency) == 2 * loss_report_frequency - 1:
                # Visualize the fixed noise samples
                gen.eval()
                dis.eval()
                samples = gen(fixed_noise)
                preview_samples(plt, samples)
                plt.show()
    return gen, dis, d_losses, g_losses

def train_v2(train_path, learning_rate, latent_dim, 
          batch_size=128, device='cpu', num_iters=10000, **kwargs):
    """
    Train a 128x128 DCGAN network.
    """

    dis = DCGANDiscriminatorV2().to(device)
    gen = DCGANGeneratorV2(nz=latent_dim).to(device)

    image_size = kwargs.get('image_size', (128, 128))
    loss_report_frequency = kwargs.get('loss_report_frequency', 100)

    train_set = CelebA(train_path, download=True, transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    train_loader_iter = iter(train_loader)

    optim_g = torch.optim.Adam(lr=learning_rate, params=gen.parameters(), betas=(0.5, 0.999))
    optim_d = torch.optim.Adam(lr=learning_rate, params=dis.parameters(), betas=(0.5, 0.999))

    # Iterate over batches, rather than epochs
    g_losses = []
    d_losses = []

    fixed_noise = torch.randn(size=(8, latent_dim), device=device, requires_grad=False)

    for iteration in tqdm(range(num_iters)):
        gen.train()
        dis.train()

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

        ## Calculate non-saturating generator loss
        gen.zero_grad()

        # Create a batch of sample noise
        noise = torch.randn(size=(input_batch_size, latent_dim), device=device)

        g_z = gen(noise)
        # Get -Ez[D(G(z))] 
        g_loss = adversarial_loss(dis(g_z), real_labels)

        g_loss.backward()
        optim_g.step()
        g_losses.append(g_loss.item() / input_batch_size)

        ## Calculate discriminator losses
        dis.zero_grad()
        # Get -Ex[logD(x)]
        d_real_loss = adversarial_loss(dis(image_batch), real_labels)
        # Get -Ez[log(1-D(G(z)))]
        d_fake_loss = adversarial_loss(dis(g_z.detach()), fake_labels)
        ## Same for pixelwise discriminator loss

        d_loss = (d_real_loss + d_fake_loss) / 2.0
        d_loss.backward()
        optim_d.step()
        d_losses.append(d_loss.item() / input_batch_size)

        if iteration % loss_report_frequency == loss_report_frequency - 1:
            print(f'Discriminator loss: {d_loss}, Generator loss: {g_loss}')
            # TODO: Validation here
            if iteration % (2 * loss_report_frequency)== 2 * loss_report_frequency - 1:
                # Visualize the fixed noise samples
                gen.eval()
                dis.eval()
                samples = gen(fixed_noise)
                preview_samples(plt, samples)
                plt.show()

    return gen, dis, d_losses, g_losses


def train_udcgan_with_cutmix(train_path, latent_dim, 
          batch_size=128, device='cpu', num_iters=10000, **kwargs):
    """
    Train a 64x64 U-DCGAN network with CutMix samples and consistancy loss.
    """

    dis = UGANDiscriminator().to(device)
    gen = DCGANGenerator(nz=latent_dim).to(device)

    image_size = kwargs.get('image_size', (64, 64))
    loss_report_frequency = kwargs.get('loss_report_frequency', 100)
    d_learning_rate = kwargs.get('d_learning_rate', 3e-4)
    g_learning_rate = kwargs.get('g_learning_rate', 1e-4)

    train_set = CelebA(train_path, download=True, transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    train_loader_iter = iter(train_loader)

    optim_g = torch.optim.Adam(lr=g_learning_rate, params=gen.parameters(), betas=(0.5, 0.999))
    optim_d = torch.optim.Adam(lr=d_learning_rate, params=dis.parameters(), betas=(0.5, 0.999))

    # Iterate over batches, rather than epochs
    g_losses = []
    d_losses = []

    fixed_noise = torch.randn(size=(8, latent_dim), device=device, requires_grad=False)

    for iteration in tqdm(range(num_iters)):
        gen.train()
        dis.train()

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
        real_mask = torch.ones(size=(input_batch_size, 3, *image_size), device=device)
        fake_mask = torch.zeros(size=(input_batch_size, 3, *image_size), device=device)

        ## Calculate non-saturating generator loss
        gen.zero_grad()

        # Create a batch of sample noise
        noise = torch.randn(size=(input_batch_size, latent_dim), device=device)

        g_z = gen(noise)

        # Cutmix sample probability
        cutmix_prob = torch.rand(1).item()
        # Start at 0, linearly scale to 0.5 after 20 iterations
        cutmix_threshold = min(0.5, iteration * (0.5 / 20.0))
        cutmix_examples = None
        masks = None
        compute_cutmix_loss = False
        if cutmix_prob > cutmix_threshold:
            compute_cutmix_loss = True
            # Maybe vectorize the mask fn if this is a bottleneck
            masks = []
            for _ in range(input_batch_size):
                mask, _ = get_cutmix_mask(image_size)
                masks.append(mask)
            masks = torch.stack(masks).unsqueeze(1).to(device)
            cutmix_examples = masks * image_batch + (1.0 - masks) * g_z.detach()


        # Get -Ez[D(G(z))] 
        g_loss_d_enc = adversarial_loss(dis(g_z)[0], real_labels)
        g_loss_d_dec = pixelwise_adversarial_loss(dis(g_z)[1], real_mask)

        g_loss = g_loss_d_enc + g_loss_d_dec

        g_loss.backward()
        optim_g.step()
        g_losses.append(g_loss.item() / input_batch_size)

        ## Calculate discriminator losses
        dis.zero_grad()
        # Get -Ex[logD(x)]
        d_real_enc_loss = adversarial_loss(dis(image_batch)[0], real_labels)
        # Get -Ez[log(1-D(G(z)))]
        d_fake_enc_loss = adversarial_loss(dis(g_z.detach())[0], fake_labels)
        ## Same for pixelwise discriminator loss
        # Get -Ex[logD(x)]
        d_real_dec_loss = pixelwise_adversarial_loss(dis(image_batch)[1], real_mask)
        # Get -Ez[log(1-D(G(z)))]
        d_fake_dec_loss = pixelwise_adversarial_loss(dis(g_z.detach())[1], fake_mask)

        # CutMix losses
        d_consistancy_loss = 0.0
        if compute_cutmix_loss:
            cutmix_out_enc, cutmix_out_dec = dis(cutmix_examples)
            d_cutmix_enc_loss = adversarial_loss(cutmix_out_enc, fake_labels)
            d_cutmix_dec_loss = pixelwise_adversarial_loss(cutmix_out_dec, masks.repeat(1, 3, 1, 1))
            d_consistancy_loss = d_cutmix_enc_loss + d_cutmix_dec_loss

        d_enc_loss = d_real_enc_loss + d_fake_enc_loss
        d_dec_loss = d_real_dec_loss + d_fake_dec_loss

        d_loss = d_enc_loss + d_dec_loss + d_consistancy_loss

        d_loss.backward()
        optim_d.step()
        d_losses.append(d_loss.item() / input_batch_size)

        if iteration % loss_report_frequency == loss_report_frequency - 1:
            print(f'Discriminator loss: {d_loss}, Generator loss: {g_loss}')
            if iteration % (2 * loss_report_frequency) == 2 * loss_report_frequency - 1:
                # Visualize the fixed noise samples
                gen.eval()
                dis.eval()
                samples = gen(fixed_noise)
                preview_samples(plt, samples)
                plt.show()
    return gen, dis, d_losses, g_losses

def train_udcgan_with_cutmix_v2(train_path, latent_dim, 
          batch_size=128, device='cpu', num_iters=10000, **kwargs):
    """
    Train a 128x128 U-DCGAN with CutMix samples and consistency loss. 
    """

    dis = UGANDiscriminatorV2().to(device)
    gen = DCGANGeneratorV2(nz=latent_dim).to(device)

    def weights_init(m):
        """DCGAN weight initialization, taken from PyTorch tutorial
        """
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    dis.apply(weights_init)
    gen.apply(weights_init)

    image_size = kwargs.get('image_size', (128, 128))
    loss_report_frequency = kwargs.get('loss_report_frequency', 100)
    d_learning_rate = kwargs.get('d_learning_rate', 3e-4)
    g_learning_rate = kwargs.get('g_learning_rate', 1e-4)
    train_split = kwargs.get('train_split', 'train')

    train_set = CelebA(train_path, download=True, split=train_split, transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    train_loader_iter = iter(train_loader)

    optim_g = torch.optim.Adam(lr=g_learning_rate, params=gen.parameters(), betas=(0.5, 0.999))
    optim_d = torch.optim.Adam(lr=d_learning_rate, params=dis.parameters(), betas=(0.5, 0.999))

    # Iterate over batches, rather than epochs
    g_losses = []
    d_losses = []

    fixed_noise = torch.randn(size=(8, latent_dim), device=device, requires_grad=False)

    for iteration in tqdm(range(num_iters)):
        gen.train()
        dis.train()

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
        real_mask = torch.ones(size=(input_batch_size, 3, *image_size), device=device)
        fake_mask = torch.zeros(size=(input_batch_size, 3, *image_size), device=device)

        ## Calculate non-saturating generator loss
        gen.zero_grad()

        # Create a batch of sample noise
        noise = torch.randn(size=(input_batch_size, latent_dim), device=device)

        g_z = gen(noise)

        # Cutmix sample probability
        cutmix_prob = torch.rand(1).item()
        # Start at 0, linearly scale to 0.5 after 20 iterations
        cutmix_threshold = min(0.5, iteration * (0.5 / 20.0))
        cutmix_examples = None
        masks = None
        compute_cutmix_loss = False
        if cutmix_prob > cutmix_threshold:
            compute_cutmix_loss = True
            # Maybe vectorize the mask fn if this is a bottleneck
            masks = []
            for _ in range(input_batch_size):
                mask, _ = get_cutmix_mask(image_size)
                masks.append(mask)
            masks = torch.stack(masks).unsqueeze(1).to(device)
            cutmix_examples = masks * image_batch + (1.0 - masks) * g_z.detach()


        # Get -Ez[D(G(z))] 
        g_loss_d_enc = adversarial_loss(dis(g_z)[0], real_labels)
        g_loss_d_dec = pixelwise_adversarial_loss(dis(g_z)[1], real_mask)

        g_loss = g_loss_d_enc + g_loss_d_dec

        g_loss.backward()
        optim_g.step()
        g_losses.append(g_loss.item() / input_batch_size)

        ## Calculate discriminator losses
        dis.zero_grad()
        # Get -Ex[logD(x)]
        d_real_enc_loss = adversarial_loss(dis(image_batch)[0], real_labels)
        # Get -Ez[log(1-D(G(z)))]
        d_fake_enc_loss = adversarial_loss(dis(g_z.detach())[0], fake_labels)
        ## Same for pixelwise discriminator loss
        # Get -Ex[logD(x)]
        d_real_dec_loss = pixelwise_adversarial_loss(dis(image_batch)[1], real_mask)
        # Get -Ez[log(1-D(G(z)))]
        d_fake_dec_loss = pixelwise_adversarial_loss(dis(g_z.detach())[1], fake_mask)

        # CutMix losses
        d_consistancy_loss = 0.0
        if compute_cutmix_loss:
            cutmix_out_enc, cutmix_out_dec = dis(cutmix_examples)
            d_cutmix_enc_loss = adversarial_loss(cutmix_out_enc, fake_labels)
            d_cutmix_dec_loss = pixelwise_adversarial_loss(cutmix_out_dec, masks.repeat(1, 3, 1, 1))
            d_consistancy_loss = d_cutmix_enc_loss + d_cutmix_dec_loss

        d_enc_loss = d_real_enc_loss + d_fake_enc_loss
        d_dec_loss = d_real_dec_loss + d_fake_dec_loss

        d_loss = d_enc_loss + d_dec_loss + d_consistancy_loss

        d_loss.backward()
        optim_d.step()
        d_losses.append(d_loss.item() / input_batch_size)

        if iteration % loss_report_frequency == loss_report_frequency - 1:
            print(f'Discriminator loss: {d_loss}, Generator loss: {g_loss}')
            if iteration % (2 * loss_report_frequency) == 2 * loss_report_frequency - 1:
                # Visualize the fixed noise samples
                gen.eval()
                dis.eval()
                samples = gen(fixed_noise)
                preview_samples(plt, samples)
                plt.show()
    return gen, dis, d_losses, g_losses
