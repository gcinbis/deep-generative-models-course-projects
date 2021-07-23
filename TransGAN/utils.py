import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

import os

import numpy as np

def train_loader(batch_size=30, img_size=28):

    transform = transforms.Compose([transforms.Resize(size=(28, 28)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=30, shuffle=True)

    return train_loader 

def valid_loader(batch_size=30, img_size=28):

    transform = transforms.Compose([transforms.Resize(size=(28, 28)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=30, shuffle=False)
    valid_loader = test_loader

    return valid_loader

def noise(n_samples, z_dim, device):
        return torch.randn(n_samples,z_dim).to(device)
    
class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr

def inits_weight(m):
        if type(m) == nn.Linear:
                nn.init.xavier_uniform(m.weight.data, 1.)


def noise(imgs, latent_dim):
        return torch.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim)))

def gener_noise(gener_batch_size, latent_dim):
        return torch.FloatTensor(np.random.normal(0, 1, (gener_batch_size, latent_dim)))

def save_checkpoint(states,is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))
