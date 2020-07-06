import torch
from numpy import prod

def PG_epsilon(data_distribution, generated):

    s = data_distribution.shape
    epsilon = torch.rand(s[0], 1)
    epsilon = epsilon.expand(s[0], prod(s[1:])).contiguous().reshape(s)
    
    return epsilon


def PG_Gradient_Penalty(data_distribution, model_distribution, discriminator, lamda=10., backward=True):

    # used GAN ZOO implementation
    epsilon = PG_epsilon(data_distribution, model_distribution).to(data_distribution.device)
    x_hat = epsilon * data_distribution + ((1.-epsilon) * model_distribution)
    x_hat = torch.autograd.Variable(x_hat, requires_grad = True)
    scalar_output = discriminator(x_hat)
    gradients = torch.autograd.grad(outputs=scalar_output[:,0].sum(), inputs=x_hat, create_graph=True, retain_graph=True)
    gradients = gradients[0].view(data_distribution.shape[0], -1)
    gradients = (gradients * gradients).sum(dim=1).sqrt()
    gradient_penalty = (((gradients - 1.0)**2.) * lamda).mean()

    if backward:
        gradient_penalty.backward(retain_graph=True)

    return gradient_penalty.item()
