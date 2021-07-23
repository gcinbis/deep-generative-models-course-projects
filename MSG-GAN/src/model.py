from torch.nn import Module, ModuleList, LeakyReLU

from torch import save

from collections import OrderedDict

from .custom import (
    GeneratorInitialBlock, 
    GeneratorBlock, 
    DiscriminatorInitialBlock,
    DiscriminatorMidBlock,
    DiscriminatorFinalBlock
    )

class Generator(Module):
    """
    Generator of the MSG-GAN. Its architecture is explained in main.ipynb.
    """
    def __init__(self, 
            num_blocks=2,
            channels=[512,512,512,512,512,256,128,64,32,16],
            kernel_size=3,
            padding=1,
            stride=1,
            initial_spatial_dim=4, 
            img_channels=3,
            activation=LeakyReLU(0.2),
            scale_factor=2,
            ):
        """
        Args:
            num_blocks: number of blocks in the model. see table 6 in the paper for the information about blocks
            channels: number of channels in blocks
            kernel_size: kernel size of the convolution operations
            padding: padding of the convolution operations
            stride: stride of the convolution operations
            initial_spatial_dim: size of the smallest model output image, which is the output of first block
            img_channels: number of image channels
            activation: activation function used in the model
            scale_factor: image size multiplier between blocks
        """
        super().__init__()
        
        err = "num blocks must be less then len(channels)."
        assert num_blocks < len(channels), err
        
        self.blocks = ModuleList()
        self.num_blocks= num_blocks
        for blk in range(num_blocks):
            self.blocks.append(
                GeneratorInitialBlock(
                    in_dimension=channels[blk],
                    spatial_dimension=initial_spatial_dim,
                    out_channels=channels[blk+1],
                    img_channels=img_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    activation=activation
                ) if blk == 0 else
                GeneratorBlock(
                    in_channels=channels[blk],
                    out_channels=channels[blk+1],
                    scale_factor=scale_factor,
                    img_channels=img_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    activation=activation
                )
            )
    
    def forward(self, x, generate_images=range(9)):
        """
        This function returns generated images of different sizes.
        """
        imgs_out = OrderedDict()

        for blk in range(self.num_blocks):
            x, imgs_out[blk] = (
                self.blocks[blk](x, generate_img=blk in generate_images)
            )

        return imgs_out
    
    def save(self, address):
        """
        This functions saves the model at the given address.
        """
        save(self, address)


class Discriminator(Module):
    """Discriminator of the MSG-GAN. 
    Its architecture is explained in main.ipynb.
    """
    def __init__(
            self, 
            num_blocks=2,
            channels=[16, 32, 64, 128, 256, 512, 512, 512, 512, 512],
            kernel_size=3,
            padding=1,
            stride=1,
            final_spatial_dim=4, 
            img_channels=3,
            activation=LeakyReLU(0.2),
            dimension_reduction=2,
            scheme="simple"
            ):
        """
        Args:
            num_blocks: number of blocks in the model. 
                see table 7 in the paper for the information about blocks
            channels: number of channels in blocks
            kernel_size: kernel size of the convolution operations
            padding: padding of the convolution operations
            stride: stride of the convolution operations
            final_spatial_dim: size of the smallest model input image, 
                which is the input of final block
            img_channels: number of image channels
            activation: activation function used in the model
            dimension_reduction: image size multiplier between blocks
            scheme: phi scheme used in the model. 
                available options are 'simple', 'lin_cat' and 'cat_lin'
        """
        super().__init__()

        err = "num blocks must be less then len(channels)."
        assert num_blocks < len(channels), err

        self.blocks = ModuleList()
        self.num_blocks= num_blocks    
        for blk in range(num_blocks-1):
            idx = blk+len(channels)-num_blocks-1
            self.blocks.append(
                DiscriminatorInitialBlock(
                    in_channel=channels[idx],
                    out_channel=channels[idx+1],
                    img_channel=img_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    activation=activation,
                    dimension_reduction=dimension_reduction
                ) if blk == 0 else 
                DiscriminatorMidBlock(
                    in_channel=channels[idx],
                    out_channel=channels[idx+1],
                    img_channel=img_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    activation=activation,
                    dimension_reduction=dimension_reduction,
                    scheme=scheme                    
                )
            )
        idx = len(channels)-2
        self.blocks.append(
            DiscriminatorFinalBlock(
                    in_channel=channels[idx],
                    out_channel=channels[idx+1],
                    img_channel=img_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    activation=activation,
                    spatial_dimension=final_spatial_dim,
                    scheme=scheme   
            )
        )

    def forward(self, img_dict):
        """
        This function returns the discriminator output.
        """
        idx = sorted(list(img_dict.keys()))[::-1]
        x = self.blocks[0](img_dict[idx[0]])
        for blk in range(1, self.num_blocks):
            x = self.blocks[blk](x, img_dict[idx[blk]])
        return x
    
    def save(self, address):
        """
        This functions saves the model at the given address.
        """
        save(self, address)