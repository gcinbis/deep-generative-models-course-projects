import matplotlib.pyplot as plt
import numpy as np

import torch
from model.generator import Generator
from model.discriminator import Discriminator
from loss.WGANGP import PG_Gradient_Penalty
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor
from os import getcwd
from numpy import array, log2, interp, linspace
from numpy.random import randn
from time import time, sleep
import matplotlib.gridspec as gridspec


config = {'channels':[128,128,128,128,128,128,128], # must be len(config['sr]) + 1 
          'latent_size':128, 
          'sr':[4, 8, 16, 32, 64, 128], # spatial resolution
          'start_sr':32,
          'level_batch_size':[256, 256, 256, 128, 64, 16],
          'epochs_before_jump':[16, 15, 15, 15, 15, 15], 
          'learning_rate_generator':0.1,
          'learning_rate_critic':0.1, 
          'generator_betas':(0.0, 0.99), 
          'critic_betas':(0.0, 0.99), 
          'ncrit':1, 
          'critic_lambda':10.,
          'epsilon_drift':0.001,
          'dataset_dir':'/home/deniz/Desktop/data_set/CelebAMask-HQ/', 
          'stat_format':'epoch {:4d} resolution {:4d} critic_loss {:6.4f} generator_loss {:6.4f} time {:6f}'}

level_index = 4
device = torch.device('cuda:0')


def show_img(d):
    plt.clf()
    h = 10
    s = d.shape[0]
    fig = plt.figure(figsize=(config['sr'][level_index], config['sr'][level_index]))
    m = int(s / h)
    ax = [plt.subplot(m+1,h,i) for i in range(1, s+1)]
    for i in range(1, s+1):
        plt.axis('on')
        ax[i-1].imshow(d[i-1,:,:], cmap='gray')
        ax[i-1].set_xticklabels([])
        ax[i-1].set_yticklabels([])
        ax[i-1].set_aspect('equal')

    fig.subplots_adjust(hspace=0, wspace=0.1)
    fig.tight_layout()
    plt.show(block=False)    
    plt.pause(10)
    plt.close()

    return


generator = Generator(config['sr'][level_index], config, transition=True, save_checkpoint=False).to(device)

x = torch.randn(20, config['latent_size']).to(device)
a = generator(x) # .reshape(3, config['sr'][level_index], config['sr'][level_index])
image = array((a).tolist()).astype(int)
image = np.transpose(image, (0,2,3,1)) 
show_img(image)


'''
# uncomment this for checking the progress of the network
while True:
    
    generator = Generator(config['sr'][level_index], config, transition=False, transition_coef=0.8,  save_checkpoint=False).to(device)

    x1 = randn(config['latent_size'])
    
    x2 = randn(config['latent_size'])

    alpha = linspace(0.,1.,40)
    d = []
    for i in alpha:
        d.append(x1 * i + x2 * (1-i))

    kk = torch.Tensor(array(d)).to(device)
    a = generator(kk) # .reshape(3, config['sr'][level_index], config['sr'][level_index])
    image = array((a).tolist()).astype(int)
    image = np.transpose(image, (0,2,3,1)) 
    show_img(image)
    print()

'''