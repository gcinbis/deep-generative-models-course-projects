import torch
from .discriminator_blocks import DiscriminatorMidBlock as DMB 
from .discriminator_blocks import DiscriminatorFinalBlock as DFB
from numpy import log2
from .custom_classes import PGfromRGB, PGDownsample
from torch.nn import Module, Sequential
from os import getcwd
from os.path import exists


class Discriminator(Module):

    def __init__(self, spatial_resolution, configuration, transition=True, transition_coef=0.5, save_checkpoint=True, 
                 load_initial_block=True, img_chanels=3, group_size=4, alpha=0.2):
        
        super(Discriminator, self).__init__()

        self.spatial_resolution = spatial_resolution
        self.minimum_spatial_resolution = 4
        self.levels = int(log2(self.spatial_resolution) - log2(self.minimum_spatial_resolution)) # starts from zero
        self.config = configuration
        self.channels = self.config['channels'][:self.levels + 2]
        self.latent_size = int(self.config['latent_size'])
        self.transition_coef = transition_coef
        self.save_checkpoint = save_checkpoint
        self.transition_level = -1 if self.levels == 0 else self.levels # todo be sure of this change not levels + 1 now 
        self.transition = transition if self.spatial_resolution != self.minimum_spatial_resolution else False
        self.load_initial_block = True if load_initial_block and exists('{0}/weights/Dis_{1}x{1}'.format(getcwd(), self.spatial_resolution)) else False
        self.group_size = group_size
        self.alpha = alpha

        self.discriminator_body = Sequential()

        assert getcwd().replace('//', '/').split('/')[-1] == 'ProGan', 'make sure you are at the directory /ProGan.\n'

        self.fromrgb = PGfromRGB(self.channels[-1], img_chanels, 0.2, self.spatial_resolution)
        fromrgb_file_name = '{1}/weights/FromRGB_{0}x{0}'.format(self.spatial_resolution, getcwd())
        self.fromrgb = torch.load(fromrgb_file_name) if exists(fromrgb_file_name) else self.fromrgb #load

        if self.spatial_resolution == self.minimum_spatial_resolution:
            self.first_block = DFB(self.channels[1], self.channels[0], self.group_size, self.alpha)
            file_name = '{0}/weights/Dis_{1}x{1}'.format(getcwd(), self.spatial_resolution)
            self.first_block = torch.load(file_name) if exists(file_name) else self.first_block

        else: 
            self.first_block = DMB(self.spatial_resolution, self.channels[self.levels], self.channels[self.levels-1], self.alpha)
            file_name = '{0}/weights/Dis_{1}x{1}'.format(getcwd(), self.spatial_resolution)
            self.first_block = torch.load(file_name) if self.load_initial_block and exists(file_name) else self.first_block #load

            if self.transition:
                # create and load the fade in block is transition
                self.fade_in = Sequential()
                self.fade_in.add_module("transition_downsample", PGDownsample())
                block = PGfromRGB(self.channels[-2], img_chanels, 0.2, self.spatial_resolution // 2)

                fromrgb_file_name = '{1}/weights/FromRGB_{0}x{0}'.format(self.spatial_resolution // 2, getcwd())
                block = torch.load(fromrgb_file_name) if exists(fromrgb_file_name) else block   #load
                self.fade_in.add_module("transition_FromRGB", block)


            for level in range(self.levels-1, -1, -1):

                # dummy variables used to initialize and load the discrminator
                width = int(2**(level + log2(self.minimum_spatial_resolution))) # width and height have the same size so use one of them 
                file_name = '{0}/weights/Dis_{1}x{1}'.format(getcwd(), width) # name of possible saved block
                
                # be careful about the channel in and out here
                block = DFB(self.channels[1], self.channels[0], self.group_size, self.alpha) if level == 0 else DMB(width, self.channels[level], self.channels[level-1], self.alpha) 
                block = torch.load(file_name) if self.exists(file_name) else block #load
                
                self.discriminator_body.add_module("block_{0}x{0}".format(width), block)
        
        if self.save_checkpoint:
            self.save()

    def forward(self, x):

        x1 = self.first_block(self.fromrgb(x))
        if self.transition:
            x1 = self.fade_in.forward(x) * (1. - self.transition_coef) + self.transition_coef * x1

        x = self.discriminator_body.forward(x1)
        return x

    def exists(self, directory):
        if exists(directory): return True
        else: print('block {} does not exist. program will continue without loading.\n'.format(directory))
        return False

    def save(self):

        self.fromrgb.save()
        self.first_block.save()

        for blocks in self.discriminator_body:
            blocks.save()
        if self.transition:
            self.fade_in[1].save()
        
        return 

'''
config = {'channels':[512,512,512,512,256,128,64,32,16], 'latent_size':512}
x = torch.randn(12, 3, 16, 16)
a = Discriminator(16, config, transition=True)
# print(a.fade_in[1]) # the blocks are exactly the same as in the paper  
print(a(x).shape)
'''