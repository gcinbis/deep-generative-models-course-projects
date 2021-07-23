from torch.functional import norm
from torch.nn import (
    Module, Upsample, 
    AvgPool2d, LeakyReLU
    )

import torch.nn as nn

from torch import (
    sqrt, var, cat, randn, Tensor, device
)
import torch.linalg
from torch.serialization import save
from numpy import array
import pickle
import os


def get_latent_variable(batchsize, latent_dimension, device):
    """Creates a random vector of size (batchsize, latent_dimension)
        from normal distribution.
    """
    latent_var = randn(
        (batchsize, latent_dimension), requires_grad=True
        ).to(device)
    return latent_var / torch.linalg.norm(latent_var, ord=2, dim=-1).reshape(latent_var.shape[0], 1)


def pixel_norm(x, epsilon=1e-8):
    """Return the pixel norm of the input.
    Input: activation of size NxCxWxH 
    """
    return x / sqrt((x**2.).mean(axis=1, keepdim=True) + epsilon)


def minbatchstd(x, group_size=2, eps=1e-8):
    """Implementation of the minbatch standard deviation.
    """
    
    x_size = x.shape
    err = "Batch size must be divisible by group size"
    assert x_size[0] % group_size == 0, err
    
    group_len = x_size[0] // group_size
    y = x.view(group_len, group_size, *x_size[1:])
    y = var(y, dim=1)
    y = torch.sqrt(y + eps)
    y = y.view(group_len, -1)
    y = y.mean(dim=1).view(group_len, 1)
    y = y.expand(group_len, x_size[2] * x_size[3])
    y = y.view(group_len, 1, 1, x_size[2], x_size[3])
    y = y.expand(-1, group_size, -1, -1, -1)
    y = y.reshape(-1, 1, x_size[2], x_size[3])
    x = cat([x, y], dim=1)
    return x


class Linear(Module):
    """Overloading the Linear layer for compatibility with paper.

    The standard pytorch implementation of Linear layer 
    uses kiaming_uniform_ to initilize the weights it is 
    modified to conform with the paper Progressive Growing of 
    GANs for Improved Quality, Stability, and Variation paper 
    for initialization.
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)
    

class Conv2d(Module):
    """Overloading the Conv2d layer for compatibility with paper.

    The standard pytorch implementation of _ConvNd layer 
    uses init.kaiming_uniform_(self.weight, a=math.sqrt(5)) 
    to initilize the weights it is modified to conform 
    with the paper Progressive Growing of GANs for Improved 
    Quality, Stability, and Variation paper for initialization.    
    """

    def __init__(
        self, in_channels, out_channels, 
        kernel_size=(3,3), 
        stride=(1,1), 
        padding=(0,0)
        ):
        super().__init__()

        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv2d.weight)
    
    def forward(self, x):
        return self.conv2d(x)


class PhiScheme(Module):
    """Defines the phi scheme described in the paper.

    phi is a function used to combine the output o_i 
    of the (i)th intermediate layer of the generator 
    (or correspondingly downsampled version of the 
    highest resolution real image y) with the 
    corresponding output of the (j−1)th 
    intermediate layer in the discriminator.

    []  -> indicates concatination
    r() -> 1x1 convolution operation
    phi_{simple}(x1, x2) = [x1; x2]
    phi_{lin-cat}(x1, x2) = [r(x1); x2]
    phi_{cat-lin}(x1, x2) = r([x1; x2])

    see: https://arxiv.org/pdf/1903.06048.pdf
    equations: 11-12-13
    """

    def __init__(self, img_channels, in_channels, scheme="simple") -> Tensor:
        super().__init__()

        self.scheme = scheme
        schemes = ["simple", "lin_cat", "cat_lin"]
        err = "Select one of {} {} {} as scheme.".format(*schemes)
        assert scheme.lower() in schemes, err

        self.r_prime = Conv2d(
                in_channels=img_channels, 
                out_channels=in_channels, 
                kernel_size=1) if scheme in schemes[1:] else None
        
    def forward(self, x1, x2):
        if self.scheme == "simple":
            return cat([x1,x2], dim=1)
        elif self.scheme == "lin_cat":
            return cat([self.r_prime(x1), x2], dim=1)
        elif self.scheme == "cat_lin":
            return self.r_prime(cat([x1, x2], dim=1))
        else:
            raise("No valid scheme is selected")
        

class FromRGB(Module):
    """Implementation of FromRGB 0 in paper page 11 Table 7
    
    see: https://arxiv.org/pdf/1903.06048.pdf 
    Table: 7

    Here FromRGB is implemented as a 1x1 convolution 
    operation to convert image channel size to the model
    required channels as described in Progressive Growing
    of GANs for Improved Quality, Stability, and Variation
    page 4 figure 2 description.

    see: https://arxiv.org/pdf/1710.10196.pdf
    Table 2. 
    """

    def __init__(self, img_channels, out_channels) -> Tensor:
        super().__init__()

        self.from_rgb = Conv2d(
            in_channels=img_channels,
            out_channels=out_channels,
            kernel_size=1
        ) 

    def forward(self, x):
        return self.from_rgb(x)


class GeneratorInitialBlock(Module):
    """Implements the initial block of the generator.

    The generator consists of two distinct block types.
    
    GeneratorInitialBlock is the initial block of the generator
    of MSG-ProGAN that converts the latent vector to the input 
    of the second block.

    Its signature is as follows:
    latent vector -> Norm -> conv4x4 -> 
    -> LReLU -> conv3x3 -> LReLU -> output

    Here instead of conv4x4 we did the following:
    
    let s = latent vector dimension
    
    latent vector -> Norm -> 
    -> Linear(N, s, s*4*4).reshape(N, s, 4, 4)

    see: https://arxiv.org/pdf/1903.06048.pdf 
    Table 6: Generator architecture for the MSG-ProGAN model.
    """

    def __init__(
            self, in_dimension=512, spatial_dimension=4, out_channels=512, 
            img_channels=3, kernel_size = (3,3), stride = (1,1), 
            padding = (0,0), activation=LeakyReLU(0.2)) -> Tensor:
        super().__init__()
        
        self.in_dimension = in_dimension
        self.spatial_dimension = spatial_dimension
        self.out_channels = out_channels
        self.img_channels = img_channels

        self.activation = activation
        self.linear = Linear(in_dimension, in_dimension * int(spatial_dimension ** 2.))
        self.conv = Conv2d(
            in_channels=in_dimension, 
            out_channels=out_channels, 
            kernel_size=kernel_size,
            stride=stride, 
            padding=padding)
        self.img_conv = Conv2d(
            in_channels=self.out_channels, 
            out_channels=self.img_channels, 
            kernel_size=1)
        
    def forward(self, x, generate_img=True):
        x = self.linear(x)
        x = x.view(-1, self.in_dimension, 
            self.spatial_dimension, self.spatial_dimension)
        x = self.activation(x)
        x = pixel_norm(self.activation(self.conv(x)))
        if generate_img:
            return x, self.img_conv(x)
        else:
            return x, None

    def extra_repr(self) -> str:
        return "in dim {0}, out_dim {1}x{2}x{2}".format(
            self.in_dimension, self.out_channels, self.spatial_dimension
        )


class GeneratorBlock(Module):
    """Implementation of the Generator blocks.

    The blocks 2 to 9 in Table 6. have the same 
    signature and are implemened in this class.

    see: https://arxiv.org/pdf/1903.06048.pdf 
    Table 6: Generator architecture for the MSG-ProGAN model.
    
    The signature of the mid blocks is as follows:
    a_{i-1} = Output of activation of previous layer.
    a_{i-1} -> Upsample -> conv3x3 -> LReLU ->
            -> conv3x3 -> LReLU -> a_{i}
    """

    def __init__(
            self, in_channels=512, out_channels=512, 
            scale_factor=2, img_channels=3, kernel_size=(3,3), 
            stride=(1,1), padding=(1,1), 
            activation=LeakyReLU(0.2)) -> Tensor:
        super().__init__()

        self.activation = activation
        self.upsample = Upsample(scale_factor=scale_factor)
        self.conv_first = Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)
        self.conv_second = Conv2d(
            in_channels=out_channels, 
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)
        self.img_conv = Conv2d(
            in_channels=out_channels, 
            out_channels=img_channels, 
            kernel_size=1)        
    def forward(self, x, generate_img=True):
        x = self.upsample(x)
        x = pixel_norm(self.activation(self.conv_first(x)))
        x = pixel_norm(self.activation(self.conv_second(x)))
        if generate_img:
            return x, self.img_conv(x)
        else:
            return x, None        


class DiscriminatorFinalBlock(Module):
    """Implementation of the final block of the discriminator.
    The discriminator consists of three distinct block types. 

    The initial and final blocks of the discriminator are 
    always present and the number of mid blocks might change
    due to the desired resolution and thus the total number 
    of blocks that are going to be used in the model.


    see: https://arxiv.org/pdf/1903.06048.pdf 
    Table 7: Discriminator architecture for the MSG-ProGAN model.

    The signature of the final block is as follows:
    
    (input, a{l-1}) -> phi_{simple} -> MinBatchStd -> 
        conv3x3 -> LReLU -> conv4x4 -> LReLU -> Linear 
        -> output

    input  -> (N, 3, 4, 4)
    a{l-1} -> (N, 515, 4, 4)
    output -> (N, 1, 1) 
    """

    def __init__(
            self, in_channel, out_channel, spatial_dimension=4,
            img_channel=3, kernel_size = (3,3), stride = (1,1), 
            padding = (1,1), activation=LeakyReLU(0.2), 
            scheme="simple") -> Tensor:
        super().__init__()

        self.activation = activation
        self.concat = PhiScheme(
            img_channels=img_channel, 
            in_channels=in_channel, 
            scheme=scheme
            )
        self.conv_first = Conv2d(
            in_channels=img_channel + in_channel + 1,
            out_channels=in_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
            )
        self.conv_second = Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=spatial_dimension, 
            padding=0
            )
        self.linear = Linear(
            in_features=out_channel,
            out_features=1)
    
    def forward(self, a_prime, o):
        x = minbatchstd(self.concat(a_prime, o))
        x = self.activation(self.conv_first(x))
        x = self.activation(self.conv_second(x))
        x_shape = x.shape
        x = x.reshape(x_shape[0], -1)
        x = self.linear(x)
        return x


class DiscriminatorMidBlock(Module):
    """Implementation of the mid block of the discriminator/critic
    The discriminator consists of three distinct block types. 

    The initial and final blocks of the discriminator are 
    always present and the number of mid blocks might change
    due to the desired resolution and thus the total number 
    of blocks that are going to be used in the model.


    see: https://arxiv.org/pdf/1903.06048.pdf 
    Table 7: Discriminator architecture for the MSG-ProGAN model.

    The signature of the final block is as follows:
    
    (input, a{l-1}) -> phi_{simple} -> MinBatchStd -> 
        conv3x3 -> LReLU -> conv3x3 -> LReLU -> AvgPool 
        -> output

    input  -> (N, 3, dim, dim)
    a{l-1} -> (N, 515, dim, dim)
    output -> (N, dim//2, dim//2)    
    """

    def __init__(
            self, in_channel, out_channel, img_channel=3, 
            dimension_reduction=2, kernel_size=(3,3), stride=(1,1), 
            padding = (1,1), activation=LeakyReLU(0.2), 
            scheme="simple") -> Tensor:
        super().__init__()

        self.activation = activation
        self.concat = PhiScheme(
            img_channels=img_channel, 
            in_channels=in_channel, 
            scheme=scheme
            )
        self.conv_first = Conv2d(
            in_channels=img_channel + in_channel + 1,
            out_channels=in_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
            )
        self.conv_second = Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
            )
        self.avg_pool = AvgPool2d(
            kernel_size=dimension_reduction, 
            stride=dimension_reduction
            )
    
    def forward(self, a_prime, o):
        x = minbatchstd(self.concat(a_prime, o))
        x = self.activation(self.conv_first(x))
        x = self.activation(self.conv_second(x))
        return self.avg_pool(x)


class DiscriminatorInitialBlock(Module):
    """Implementation of the initial block of the discriminator/critic
    The discriminator consists of three distinct block types. 

    The initial and final blocks of the discriminator are 
    always present and the number of mid blocks might change
    due to the desired resolution and thus the total number 
    of blocks that are going to be used in the model.


    see: https://arxiv.org/pdf/1903.06048.pdf 
    Table 7: Discriminator architecture for the MSG-ProGAN model.

    The signature of the final block is as follows:
    
    img -> FromRGB 0 -> MinBatchStd -> conv3x3 -> LReLU 
        -> conv3x3 -> LReLU -> AvgPool -> output

    input  -> (N, 3, dim, dim)
    output -> (N, dim//2, dim//2)  
    
    """

    def __init__(
            self, in_channel, out_channel, img_channel=3, 
            dimension_reduction=2, kernel_size=(3,3), stride=(1,1), 
            padding = (1,1), activation=LeakyReLU(0.2)) -> Tensor:
        super().__init__()

        self.activation = activation
        self.from_rgb = FromRGB(
            img_channels=img_channel,
            out_channels=in_channel
            )
        self.conv_first = Conv2d(
            in_channels=in_channel + 1,
            out_channels=in_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
            )
        self.conv_second = Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
            )
        self.avg_pool = AvgPool2d(
            kernel_size=dimension_reduction, 
            stride=dimension_reduction
            )
    
    def forward(self, o):
        x = minbatchstd(self.from_rgb(o))
        x = self.activation(self.conv_first(x))
        x = self.activation(self.conv_second(x))
        return self.avg_pool(x)


class LossTracker:
    """Tracks the loss
    Adds the statistics of the loss each num_iters times
    to loss_tracker list. Saves the results to a pickle 
    each num_iters * save_iters times.
    """

    def __init__(
            self, num_iters=100, 
            save_iters=100, 
            eps=1e-4,
            save_dir=f"{os.getcwd()}/stats",
            save_name="discriminator_loss"
            ):
        self.num_iters = num_iters
        self.save_iters = save_iters
        self.eps = eps
        self.save_dir = save_dir
        self.save_name = save_name
        self.loss_tracker = []
        self.tracker = []
        self.current = 0.
        self.previous = 100.

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    
    def append(self, x):
        self.current = x
        self.tracker.append(x)
        if (
            len(self.tracker) == self.num_iters
            or (len(self.tracker) == 1 and len(self.loss_tracker) == 0)
            ):
            tracker = array(self.tracker)
            self.loss_tracker.append([
                tracker.mean(),
                tracker.min(),
                tracker.max(),
                tracker.std()
                ])
            self.tracker = []
        if len(self.loss_tracker) % (self.save_iters)  == 0:
            with open(f"{self.save_dir}/{self.save_name}.pickle", "wb") as fid:
                pickle.dump(self.loss_tracker, fid)
    
    def converged(self):
        if abs(self.current - self.previous) < self.eps:
            return True
        else:
            self.previous = self.current
            return False