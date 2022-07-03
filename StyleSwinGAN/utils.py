from ast import arg
from xmlrpc.client import getparser
import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np


def generate_img(args, generator, isNorm=True):
    '''
    Generates a single img given generator.
    '''
    # Sample an image
    style_dim = args.style_dim
    noise = torch.randn((1, style_dim)).to(args.device)
    img1 = generator(noise)

    if not args.ourGen: # their model returns also the latent space, but we dont
        img1 = img1[0]
    
    if isNorm: # if img is normalized => denormalize
        img1 = tensor_transform_reverse(img1)

    img = img1[0].cpu().permute(1,2,0).detach()
    plt.imshow(img)
    plt.axis('off')


def generate_img_grid(args, generator, nrows_ncols=(5,4), isNorm=True):
    '''
    Generates image grid of shape nrows_ncols similar to figure-12 and figure-13 in the paper.
    '''
    img_list = []
    for i in range(20):
        style_dim = args.style_dim
        noise = torch.randn((1, style_dim)).to(args.device)
        img1 = generator(noise)

        if not args.ourGen: # their model returns also the latent space, but we dont
            img1 = img1[0]
        
        if isNorm:
            img1 = tensor_transform_reverse(img1)

        img = img1[0].cpu().permute(1,2,0).detach()
        img_list.append(img)

    fig = plt.figure(figsize=(100., 100.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=nrows_ncols,  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    for ax, im in zip(grid, img_list):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
    plt.show()

def create_generator(args):
    '''
    Given arguments for a model configuration, creates the generator.
    '''

    try:
        args.size = args.resolution
    except AttributeError:
        args.resolution = args.size
        args.device = 'cuda:0'

    if args.ourGen:
        from models.ours.generator_ours import Generator
        generator = Generator(
            dim=args.dim,
            style_dim=args.style_dim,
            n_style_layers=args.n_style_layers,
            n_heads=args.n_heads,
            resolution=args.size,
            attn_drop=args.attn_drop
        )
    else:
        from models.original.generator import Generator
        args.n_mlp = 8 
        args.G_channel_multiplier = 1
        args.enable_full_resolution = 8
        args.use_checkpoint = 0

        generator = Generator(
                args.size, args.style_dim, args.n_mlp, channel_multiplier=args.G_channel_multiplier, lr_mlp=args.lr_mlp,
                enable_full_resolution=args.enable_full_resolution, use_checkpoint=args.use_checkpoint
            )
    return generator.to(args.device)

def create_discriminator(args):
    '''
    Given arguments for a model configuration, creates the discriminator.
    '''

    args.size = args.resolution
    args.D_channel_multiplier = 1

    if args.ourDisc:
                if args.D_sn:
                    from models.ours.discriminator_ours_spectral import Discriminator
                else:
                    from models.ours.discriminator_ours import Discriminator
                discriminator = Discriminator(resolution=args.size)
    else:
        from models.original.discriminator import Discriminator
        discriminator = Discriminator(args.size, channel_multiplier=args.D_channel_multiplier, sn=args.D_sn)
    return discriminator.to(args.device)


def tensor_transform_reverse(image):
    ''' 
    un-normalize lsun (shift-scale) transformation 
    '''
    assert image.dim() == 4
    moco_input = torch.zeros(image.size()).type_as(image)
    moco_input[:,0,:,:] = image[:,0,:,:] * 0.229 + 0.485
    moco_input[:,1,:,:] = image[:,1,:,:] * 0.224 + 0.456
    moco_input[:,2,:,:] = image[:,2,:,:] * 0.225 + 0.406
    return moco_input


def discriminator_loss(real_pred, fake_pred):
    """
    Softplus Loss will be used for Discriminator Logistic Loss:
        >> Softplus(x) = ln(1 + exp(x))
    """
    fake_loss = F.softplus(fake_pred)
    real_loss = F.softplus(-real_pred)

    return fake_loss.mean() + real_loss.mean()


def generator_loss(fake_pred):
    """
    Generator Loss is the same function as Discriminator loss: Softplus
    """
    gen_loss = F.softplus(-fake_pred).mean()
    return gen_loss


def adjust_gradient(model, req_grad):
    """
    Adjusts the gradient changes required for training Discriminator
    and Generator.
    """
    # Change the model parameters `requires_grad` parameter to input flag
    acc_grad = 0
    for parameter in model.parameters():
        parameter.requires_grad = req_grad
        if parameter.grad is not None:
            acc_grad += (parameter.grad.clone()).mean()


def gradient_penalty(real_pred, real_img):
    """
    Also called R1 Loss. It is used in Discriminator as a regularization.
    Takes Real images and prediction for real images.
    Returns the sum of squared gradients.
    """
    

    grad_real, = torch.autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty
