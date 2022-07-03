import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

                                 
class Discriminator(nn.Module):
    def __init__(self, n_activ_maps=128, n_channels = 3, resolution=128):
        super().__init__()
        
        n_layers = int(math.log2(resolution)-1)
        
        
        net =   [
                    [
                        spectral_norm(nn.Conv2d(n_channels, n_activ_maps, 4, 2, 1, bias=False)),
                        nn.LeakyReLU(0.2, inplace=True)
                    ]
                ] +\
                [
                    [
                        spectral_norm(nn.Conv2d(n_activ_maps*(2**i), n_activ_maps * (2**(i+1)), 4, 2, 1, bias=False)),
                        nn.BatchNorm2d(n_activ_maps * (2**(i+1))),
                        nn.LeakyReLU(0.2, inplace=True),
                    ] for i in range(n_layers-2)
                ] +\
                [
                    [
                        spectral_norm(nn.Conv2d(n_activ_maps*(2**(n_layers-2)), 1, 4, 1, 0, bias=False)),
                        nn.LeakyReLU(0.2)
                    ]
                ]
        net = sum(net, [])
        self.disc = nn.Sequential(*net)

    def forward(self, input):
        return self.disc(input)