import torch.nn as nn 
import math
from models.ours.basic_layers_ours import *

class BasicBlock(nn.Module):
    '''
    High level block that is stacked in the generator/

    Args:
        dim: dimension of Height and Width
        style_dim: dimension of style

    Output:
        x_up: upsampled signal from StyleSwinTransformerBlock
        x_trgb: rgb-ed signal from StyleSwinTransformerBlock
    '''
    def __init__(self, 
               dim, 
               channel_dim,
               channel_dim_out, 
               style_dim, 
               n_heads,
               window_size=8,
               attn_drop=0):
        super().__init__()

        self.dim = dim
        self.channel_dim = channel_dim
        self.n_heads = n_heads
        
        self.spe = SinusoidalPositionalEmbedding(
            embedding_dim=self.channel_dim//2, 
            padding_idx=0, 
            init_size=self.channel_dim//2
            )

        self.swin_block = StyleSwinTransformerBlock(
                                    dim = self.channel_dim,
                                    input_resolution = (self.dim, self.dim),
                                    num_heads = self.n_heads,
                                    style_dim=style_dim,
                                    window_size = window_size
                                    )
        
        self.trgb = tRGB(channel_dim)
        self.up = Upsample(permute=True)
        self.norm = nn.LayerNorm(channel_dim)
        self.proj = nn.Linear(channel_dim, channel_dim_out)
        
    def forward(self, x, style):
        B, H, W, C = x.shape                # B x H x W x C
        spe = self.spe.make_grid2d(H, W, B) # B x C x H x W
        spe = spe.permute(0, 2, 3, 1)       # B x H x W x C
        x = x + spe
        
        x = x.reshape(B, -1, C)             # B x L x C => swin_block expected form
        x = self.swin_block(x, style)       # B x L x C => swin_block output form
        
        x_up = x.reshape(B, H, W, C)            # B x H x W x C => upsample expected form
        x_up = self.up(x_up)                    # B x 2H x 2W x C => upsample output form
        x_up = self.norm(x_up)  # layer normalization applied
        x_up = self.proj(x_up)  # channel_dim => channel_dim_out
        
        x_trgb = self.trgb(x)                   # B x C x H x W => signal fed to upsampling pipeline
                                                # which expects input in this form => B x C x 2H x 2W
        return x_up, x_trgb

class Generator(nn.Module):
    '''
    Args:
        dim: dimension of height and width, assuming [dim x dim] => [4 x 4]
        style_dim: dimension of style
        n_style_layers: mlp pipeline size for noise-massage ie. style network.
    '''
    def __init__(self, 
                dim, 
                style_dim, 
                n_style_layers,
                n_heads,
                resolution, 
                attn_drop=0):
        super().__init__()
        # # resolution check: exponent of 2 requirement
        # assert math.log(resolution, 2) == int(math.log(resolution, 2)), "Output resolution must be power of 2!" 
        # self.resolution = resolution

        self.dim = dim
        self.n_heads = n_heads
        self.style_net = StyleMassage(n_layers=n_style_layers, style_dim=style_dim)
        
        # # Number of needed Swin Transformers to obtain `resolution` at the end.
        # # dim can start from 4, 8, 16 etc. in order to adjust number of layers and cut from train time
        exp_fact = resolution/self.dim # ie. 256/4 = 2^8/2^2 = 2^6
        self.n_transformers = int(math.log(exp_fact, 2)) + 1 # log2^6 + 1 = 7 since initial +1 layer to start upsampling

        self.channel_in_config = None
        self.channel_out_config = None
        self.head_config = None
        self.window_config = None

        if resolution==256:
            self.channel_in_config  =   [512, 512, 512, 512, 512, 256, 128]
            self.channel_out_config =   [512, 512, 512, 512, 256, 128, 512]
            self.window_config      =   [4, 8, 8, 8, 8, 8, 8]
            self.head_config        =   [16, 16, 16, 16, 16, 8, 4]
            self.dims               =   [4, 8, 16, 32, 64, 128, 256]

        elif resolution==1024:
            self.channel_in_config  =   [512, 512, 512, 512, 256, 128, 64, 32, 16]
            self.channel_out_config =   [512, 512, 512, 256, 128, 64, 32, 16, 512] # 512 not shown in table-8
            self.window_config      =   [4, 8, 8, 8, 8, 8, 8, 8, 8]
            self.head_config        =   [16, 16, 16, 16, 8, 4, 4, 4, 4]
            self.dims               =   [4, 8, 16, 32, 64, 128, 256, 512, 1024]
        
        elif resolution==128:
            self.channel_in_config  =   [512, 256, 128, 64, 32, 16]
            self.channel_out_config =   [256, 128, 64, 32, 16, 8]
            self.window_config      =   [4, 8, 8, 8, 8, 8]
            self.head_config        =   [16, 16, 16, 16, 16, 16]
            self.dims               =   [4, 8, 16, 32, 64, 128]

        elif resolution==64:
            self.channel_in_config  =   [512, 256, 128, 64, 32]
            self.channel_out_config =   [256, 128, 64, 32, 16]
            self.window_config      =   [4, 8, 8, 8, 8]
            self.head_config        =   [16, 16, 16, 16, 16]
            self.dims               =   [4, 8, 16, 32, 64]

        elif resolution==32:
            self.channel_in_config  =   [512, 256, 128, 64]
            self.channel_out_config =   [256, 128, 64, 32]
            self.window_config      =   [4, 8, 8, 8]
            self.head_config        =   [16, 16, 16, 16]
            self.dims               =   [4, 8, 16, 32]

        assert self.dims[0] == self.dim

        # basic-block network
        self.basic_blocks = nn.ModuleList()
        # up network for image generation
        self.up_blocks = nn.ModuleList()

        self.tabula_rasa = ConstantInput(dim, self.channel_in_config[0])
    
        for i in range(self.n_transformers):
            self.basic_blocks.append(BasicBlock(
                                        dim=self.dims[i],
                                        channel_dim=self.channel_in_config[i],
                                        channel_dim_out=self.channel_out_config[i],
                                        window_size=self.window_config[i],
                                        style_dim=style_dim,
                                        n_heads=self.head_config[i],
                                        attn_drop=attn_drop
                                    ))

            self.up_blocks.append(Upsample(permute=False))

    def forward(self, noise):
        # Extract style
        style = self.style_net(noise)

        # Get constant tabula input for the generator
        batch_size = noise.size(0)
        input = self.tabula_rasa(batch_size)
        B, H, W, C = input.shape
        
        # prop 1st basic-block
        x_up, x_trgb = self.basic_blocks[0](input, style)
        gen = x_trgb # generative signal

        # print('gen propg')
        for i in range(1, self.n_transformers):
            bblock = self.basic_blocks[i]
            up = self.up_blocks[i]
            gen = up(gen)
            x_up, x_trgb = bblock(x_up, style)
            gen += x_trgb
            # print(f'layer: {i} -- neg: {(gen<0).sum()} -- shape: {gen.shape} -- perc: {(gen<0).sum()/torch.ones_like(gen).sum()}')
        return gen
