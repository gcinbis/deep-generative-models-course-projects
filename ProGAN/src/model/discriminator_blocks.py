import torch
from torch.nn import Module, Sequential
from numpy import log2, sqrt
from .custom_classes import PGMiniBatchStddevLayer, PGConv, PGLeakyReLU, PGPixelNorm, PGDownsample, PGLinear
from os import getcwd


class DiscriminatorMidBlock(Module):
    
    def __init__(self, spatial_resolution, channel_in, channel_out, alpha=0.2):
        super(DiscriminatorMidBlock, self).__init__()
        
        self.spatial_resolution = spatial_resolution
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.alpha = alpha
        self.block_level = int(log2(spatial_resolution)-2)

        self.first_conv_block = Sequential()
        self.first_conv_block.add_module("conv", PGConv(self.channel_in, self.channel_in,  kernel_size=3, padding=1))
        self.first_conv_block.add_module("LeakyReLU", PGLeakyReLU(alpha=self.alpha))
        # self.first_conv_block.add_module("norm", PGPixelNorm())

        self.second_conv_block = Sequential()
        self.second_conv_block.add_module("conv", PGConv(self.channel_in, self.channel_out, kernel_size=3, padding=1))
        self.second_conv_block.add_module("LeakyReLU", PGLeakyReLU(alpha=self.alpha))
        # self.second_conv_block.add_module("norm", PGPixelNorm())

        self.downsample_block = PGDownsample()

        
    def forward(self, x):
        
        x = self.first_conv_block.forward(x)
        x = self.second_conv_block.forward(x)
        x = self.downsample_block(x)
        
        return x
        
    def save(self):
        torch.save(self, '{1}/weights/Dis_{0}x{0}'.format(self.spatial_resolution, getcwd()))
        return 'Discriminator block Dis_{0}x{0} is saved.\n'.format(self.spatial_resolution)


class DiscriminatorFinalBlock(Module):
    
    def __init__(self, channel_in, channel_out, group_size=4, alpha=0.2):
        super(DiscriminatorFinalBlock, self).__init__()
        
        self.spatial_resolution = 4
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.group_size = group_size
        self.alpha = alpha
        
        self.minibatch_std_block = PGMiniBatchStddevLayer(group_size=self.group_size)
        self.conv3x3 = Sequential()
        self.conv3x3.add_module("conv3x3", PGConv(self.channel_in + 1, self.channel_out, 3, padding=1))
        self.conv3x3.add_module("LeakyReLU", PGLeakyReLU(alpha=self.alpha))
        # self.conv3x3.add_module("PixelNorm", PGPixelNorm())

        self.conv4x4 = Sequential() 
        self.conv4x4.add_module("conv4x4", PGConv(self.channel_out, self.channel_out, 4, padding=0))
        self.conv4x4.add_module("LeakyReLU", PGLeakyReLU(alpha=self.alpha))
        # self.conv4x4.add_module("PixelNorm", PGPixelNorm())

        self.Linear_block = PGLinear(self.channel_out, 1) 

        
    def forward(self, x):
        
        x = self.minibatch_std_block(x)
        x = self.conv3x3.forward(x)
        x = self.conv4x4.forward(x)
        x = x.view(-1, self.channel_out)
        x = self.Linear_block(x)
    
        return x.view(-1,1,1,1)
        
    def save(self):
        torch.save(self, '{1}/weights/Dis_{0}x{0}'.format(self.spatial_resolution, getcwd()))
        return 'Discriminator block Dis_{0}x{0} is saved.\n'.format(self.spatial_resolution)

