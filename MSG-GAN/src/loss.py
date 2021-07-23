from torch import Tensor, rand, randn, ones_like, cat, ones
from torch.linalg import norm
from torch.autograd import grad
import numpy as np
from .model import Discriminator
from collections import OrderedDict

def gradient_penalty_loss(discriminator, from_real, from_fake):
    """Computes the gradient penalty in the WGAN-GP loss. 
    Since MSG-GAN computes the loss using different sized 
    versions of the same image, gradient penalty is 
    computed seperately for each size and the return 
    value is their average.
    """
    epsilon = rand(
        size=(
            len(from_real.keys()), 
            from_real[0].shape[0], 
            *np.ones(len(from_real[0].shape) - 1
            ).astype(int)),
        device=from_real[0].device, requires_grad=True
    )
    
    x_hat = OrderedDict()

    for layer in range(discriminator.num_blocks):
        x_hat[layer] = (
            epsilon[layer] * from_real[layer]
            + (1-epsilon[layer]) * from_fake[layer]
            ).requires_grad_(True)
    dis_out = discriminator(x_hat).sum()
    grads = grad(
        dis_out, 
        [x_hat[i] for i in x_hat.keys()], 
        create_graph=True,
        retain_graph=True
        )

    output = cat(
        [((
            norm(i.reshape(from_real[0].shape[0], -1), ord=2, dim=1) 
            - ones(from_real[0].shape[0], requires_grad=True, device=from_real[0].device)
            ) ** 2.).unsqueeze(1) for i in grads], 1)
    
    output = output.sum(dim=0).mean()
    return output


def WGANGP_loss(discriminator, from_real, from_fake, lamda=10.):
    """
    Computes the WGAN-GP loss published in 'Improved Training of Wasserstein GANs' 
    (arxiv.org/abs/1704.00028).
    Args:
        discriminator: discriminator model
        from_real: real images
        from_fake: generated images
        lamda: coefficient of the gradient penalty loss
    Returns:
        WGAN-GP loss value
    """
    return (
        discriminator(from_fake).mean()
        - discriminator(from_real).mean() 
        + lamda * gradient_penalty_loss(discriminator, from_real, from_fake)
        )   




