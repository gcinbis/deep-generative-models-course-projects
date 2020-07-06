import torch
from torch.nn import Module, Sequential
from numpy import log2, sqrt
from .custom_classes import PGLinear, PGLeakyReLU, PGConv, PGPixelNorm, PGUpsample
from os import getcwd

class GeneratorFirstBlock(Module):
    
    def __init__(self, latent_size, spatial_resolution=4, alpha=0.2):
        super(GeneratorFirstBlock, self).__init__()
        
        assert spatial_resolution == 4, "the spatial_resolution must be 4"
        
        self.spatial_resolution = spatial_resolution
        self.latent_size = latent_size
        self.alpha = alpha
        self.block_level = int(log2(spatial_resolution) - 2)
        self.channel_out = self.latent_size

        # to be consistent with the paper we use the same names as are in paper sub-block layers for this block
        self.conv4x4 = Sequential() 
        self.conv4x4.add_module("Linear", PGLinear(latent_size, latent_size * spatial_resolution * spatial_resolution, gain=sqrt(2.)))
        self.conv4x4.add_module("LeakyReLU", PGLeakyReLU(alpha=self.alpha))

        self.conv3x3 = Sequential()
        self.conv3x3.add_module("Conv3x3", PGConv(self.latent_size, self.latent_size, kernel_size=3, padding=1))
        self.conv3x3.add_module("LeakyReLU", PGLeakyReLU(alpha=self.alpha))
        self.conv3x3.add_module("PixelNorm", PGPixelNorm())
        
        
    def forward(self, x):
        x = self.conv4x4.forward(x)
        x = x.view(-1, self.latent_size, self.spatial_resolution, self.spatial_resolution)
        x = self.conv3x3.forward(x)
        return x
    
    def save(self):
        torch.save(self, '{1}/weights/Gen_{0}x{0}'.format(self.spatial_resolution, getcwd()))
        return 'Generator block Gen_{0}x{0} is saved.\n'.format(self.spatial_resolution)


class GeneratorMidBlock(Module):
    
    def __init__(self, spatial_resolution, channel_in, channel_out, transition=False, alpha=0.2):
        super(GeneratorMidBlock, self).__init__()
        
        self.spatial_resolution = spatial_resolution
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.alpha = alpha
        self.transition = transition
        self.block_level = int(log2(spatial_resolution) - 2)

        self.upsample_block = PGUpsample()

        self.first_conv_block = Sequential()
        self.first_conv_block.add_module("conv", PGConv(self.channel_in, self.channel_out, kernel_size=3, padding=1))
        self.first_conv_block.add_module("LeakyReLU", PGLeakyReLU(alpha=self.alpha))
        self.first_conv_block.add_module("norm", PGPixelNorm())

        self.second_conv_block = Sequential()
        self.second_conv_block.add_module("conv", PGConv(self.channel_out, self.channel_out, kernel_size=3, padding=1))
        self.second_conv_block.add_module("LeakyReLU", PGLeakyReLU(alpha=self.alpha))
        self.second_conv_block.add_module("norm", PGPixelNorm())
        
    def forward(self, x):
        
        x1 = self.upsample_block.forward(x)
        x = self.first_conv_block.forward(x1)
        x = self.second_conv_block.forward(x)

        # the fadein can be handled after invoking forward as well. But handelling it here seems better  

        if self.transition:
            return x1, x
        else:
            return x
    
    def save(self):
        torch.save(self, '{1}/weights/Gen_{0}x{0}'.format(self.spatial_resolution, getcwd()))
        return 'Generator block Gen_{0}x{0} is saved.\n'.format(self.spatial_resolution)
