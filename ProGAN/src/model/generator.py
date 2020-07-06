import torch
from .generator_blocks import GeneratorFirstBlock as GFB
from .generator_blocks import GeneratorMidBlock as GMB
from .custom_classes import PGtoRGB
from torch.nn import Module, Sequential, ReLU
from numpy import log2, sqrt
from os import getcwd
from os.path import exists


class Generator(Module):


    def __init__(self, spatial_resolution, configuration, transition=True, transition_coef=0.5, save_checkpoint=True, load_final_block=True):
        super(Generator, self).__init__()

        self.spatial_resolution = spatial_resolution
        self.config = configuration
        self.channels = self.config['channels']
        self.latent_size = int(self.config['latent_size'])
        self.minimum_spatial_resolution = 4
        self.transition_coef = transition_coef
        self.save_checkpoint = save_checkpoint
        self.levels = int(log2(self.spatial_resolution) - log2(self.minimum_spatial_resolution)) # starts from zero
        self.transition = transition if self.spatial_resolution != self.minimum_spatial_resolution else False
        self.transition_level = -1 if self.levels == 0 or not self.transition else self.levels # todo be sure of this change not levels + 1 now 
        self.load_final_block = True if load_final_block and exists('{0}/weights/Gen_{1}x{1}'.format(getcwd(), self.spatial_resolution)) else False

        self.generator_body = Sequential()
        self.torgb_block = PGtoRGB(self.channels[self.levels], spatial_resolution=self.spatial_resolution)
        
        assert getcwd().replace('//', '/').split('/')[-1] == 'ProGan', 'make sure you are at the directory /ProGan.\n'

        for level in range(self.levels + 1): # + 1 is added to include the final block as well
            
            # dummy variables used to initialize and load the generator
            width = int(2**(level + log2(self.minimum_spatial_resolution))) # width and height have the same size so use one of them 
            file_name = '{0}/weights/Gen_{1}x{1}'.format(getcwd(), width) # name of possible saved block
            first_level_condition = self.spatial_resolution == self.minimum_spatial_resolution

            if level != self.transition_level: # note that it includes levels=0 and transition=False see self.transition_level
                # initialize
                block = GMB(width, self.channels[level-1], self.channels[level], transition=False) if level !=0 else GFB(self.latent_size, width)

                # load
                block = torch.load(file_name) if self.exists(file_name) else block

                if level != 0:
                    block.transition = False

                # x = (torch.zeros(1, 128, 8, 8) + 0.1)
                torgb_file_name = '{1}/weights/ToRGB_{0}x{0}'.format(width, getcwd())
                loading_condition = first_level_condition or (not self.transition and level == self.levels)
                self.torgb_block = torch.load(torgb_file_name) if loading_condition and exists(torgb_file_name) else self.torgb_block



            else:
                # initialize
                block = GMB(width, self.channels[level-1], self.channels[level], transition=True)
                self.transition_torgb = PGtoRGB(self.channels[level], spatial_resolution=int(width//2))
                
                # load
                torgb_file_name = '{1}/weights/ToRGB_{0}x{0}'.format(width//2, getcwd())
                self.transition_torgb = torch.load(torgb_file_name) if exists(torgb_file_name) else self.transition_torgb

                if self.load_final_block:
                    block = torch.load(file_name) if self.exists(file_name) else block
                    block.transition = True
                    torgb_file_name = '{1}/weights/ToRGB_{0}x{0}'.format(width, getcwd())
                    self.torgb_block = torch.load(torgb_file_name) if exists(torgb_file_name) else self.torgb_block
            
            self.generator_body.add_module("block_{0}x{0}".format(width), block)
        
        if self.save_checkpoint:
            self.save()


    def forward(self, x):

        if self.transition and self.minimum_spatial_resolution != self.spatial_resolution:
            x1, x = self.generator_body.forward(x)
            x = self.transition_coef * self.torgb_block(x) + (1.-self.transition_coef) * self.transition_torgb(x1)
        else:
            x = self.generator_body.forward(x)
            x = self.torgb_block(x)
        return x


    def save(self):

        for blocks in self.generator_body:
            blocks.save()
        self.torgb_block.save()
        if self.transition:
            self.transition_torgb.save()
        return 


    def exists(self, directory):
        if exists(directory): return True
        else: print('block {} does not exist. program will continue without loading.\n'.format(directory))
        return False

