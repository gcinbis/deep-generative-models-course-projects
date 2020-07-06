import torch
from numpy import sqrt, prod
from torch import zeros_like, zeros
from torch.nn import Module, Linear, Conv2d, LeakyReLU, Upsample, AvgPool2d, Sequential
from torch.nn.init import normal_
from os import getcwd


class PGLeakyReLU(Module):
    
    def __init__(self, alpha=0.2):
        super(PGLeakyReLU, self).__init__()
        self.alpha = alpha
        self.activation = LeakyReLU(negative_slope=alpha)
        
        
    def forward(self, x):
        return self.activation(x)


class PGPixelNorm(Module):
    
    def __init__(self, epsilon=1e-8):
        super(PGPixelNorm, self).__init__()
        self.epsilon = epsilon
        
    def forward(self, x):
        return x / torch.sqrt(((x**2).mean(axis=1, keepdim=True) + self.epsilon))


class PGMiniBatchStddevLayer(Module):

    # added from GAN ZOO
    
    def __init__(self, group_size=2, epsilon=1e-8):
        super(PGMiniBatchStddevLayer, self).__init__()
        self.group_size = group_size
        self.epsilon = epsilon

    def forward(self, x):
        size = x.size()
        subGroupSize = min(size[0], self.group_size)
        assert x.shape[0] % self.group_size == 0, "batch size is not divisible by groupsize" 
        G = int(size[0] / subGroupSize)
        y = x.view(-1, subGroupSize, size[1], size[2], size[3])
        y = torch.var(y, 1)
        y = torch.sqrt(y + 1e-8)
        y = y.view(G, -1)
        y = torch.mean(y, 1).view(G, 1)
        y = y.expand(G, size[2]*size[3]).view((G, 1, 1, size[2], size[3]))
        y = y.expand(G, subGroupSize, -1, -1, -1)
        y = y.contiguous().view((-1, 1, size[2], size[3]))
        return torch.cat([x, y], dim=1)


class PG_bias(Module):

    def __init__(self, size):
        super(PG_bias,self).__init__()
        self.bias = zeros(size, requires_grad=True)

    def forward(self, x):
        return x + self.bias.to(x.device)
    
    def extra_repr(self):
        return 'bias'


class _linear(Module):

    def __init__(self, dim_in, dim_out, gain=sqrt(2.)):
        super(_linear, self).__init__()

        self.linear = Linear(dim_in, dim_out)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.gain = gain
        self.c = 1

        self.reset_parameters()

    def reset_parameters(self):
        normal_(self.linear.weight, 0.,1.)
        self.linear.bias = None
        self.c = self.gain / sqrt(self.dim_in)
        return

    def forward(self, x):
        return self.linear(x * self.c)

class PGLinear(Module):
    
    """     
        This is the linear layer that takes latent variables and create the [N, out_channel, 4, 4] inputs of generators conv 4x4 in first block.
        This layer is initialized according to the paper using Kiaming initializer. 
        Note that torch.nn.init is only used for generating weights from a normal distribution.  	
    """
    


    def __init__(self, dim_in, dim_out, gain=sqrt(2.)):

        """ Initialize PGLinear.

            :type dim_in: int
            :param dim_in: size of input samples 

            :type dim_out: int
            :param dim_out: size of output samples

            :type gain: real
            :param gain: gain used in He initializer

            :raises:

            :rtype:
        """
        
        super(PGLinear, self).__init__()
        
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.gain = gain

        # create a fully connected layer
        self.linear = Sequential()
        self.linear.add_module('linear', _linear(self.dim_in, self.dim_out, self.gain))
        self.linear.add_module('bias', PG_bias(self.dim_out))
        
        return

    def forward(self, x):
        return self.linear.forward(x)


class _conv2d(Module):

    def __init__(self, channel_in, channel_out, kernel_size=3, padding=1, stride=1, gain=sqrt(2.)):
        super(_conv2d, self).__init__()

        self.channel_in = channel_in
        self.channel_out = channel_out
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride 
        self.gain = gain

        self.fan_in = 1
        self.c = 1.

        self.conv = Conv2d(channel_in, channel_out, kernel_size=kernel_size, stride=stride, padding=padding)
        self.reset_parameters()

    def reset_parameters(self):
        
        # initialize the weights
        normal_(self.conv.weight, 0.,1.)
        # make bias None
        self.conv.bias = None
        
        # calculate per layer normalization constant
        # for (c_in, c_out, k, k) we get W.shape = (c_out, c_in, k, k)
        # He et al page 4; fan_in = in_channel * kernel_size ** 2.
        self.fan_in = prod(self.conv.weight.shape[1:]) 
        self.c = self.gain / sqrt(self.fan_in)

        return
    
    def forward(self, x):
        return self.conv(x * self.c)

class PGConv(Module):
    
    
    def __init__(self, channel_in, channel_out, kernel_size=3, padding=1, stride=1, gain=sqrt(2.)):
        
        super(PGConv, self).__init__()

        self.channel_in = channel_in
        self.channel_out = channel_out
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride 
        self.gain = gain
        
        self.conv = Sequential()
        self.conv.add_module('conv', _conv2d(self.channel_in, self.channel_out, self.kernel_size, self.padding, self.stride, self.gain))
        self.conv.add_module('bias', PG_bias((1,self.channel_out,1,1)))
        
        return
    
    def forward(self, x):
        return self.conv.forward(x)


class PGUpsample(Module):
    
    def __init__(self, scale_factor=2):
        super(PGUpsample, self).__init__()
        self.scale_factor = scale_factor
        self.upsample = Upsample(scale_factor=scale_factor)
        
    def forward(self,x):
        return self.upsample(x)


class PGDownsample(Module):
    
    def __init__(self, kernel_size=2, stride=2):
        super(PGDownsample, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        self.downsample = AvgPool2d(int(kernel_size), int(stride))

    def forward(self, x):
        return self.downsample(x)


class PGtoRGB(Module):
    
    def __init__(self, output_c, img_c=3, alpha=0.2, spatial_resolution=4):
        super(PGtoRGB, self).__init__()

        self.out_c  = output_c
        self.img_c = img_c
        self.spatial_resolution = spatial_resolution
        self.alpha = alpha
        # add a 2d convolution layer then add LeakyReLU
        self.conv = PGConv(output_c, img_c, 1, padding=0)
        self.act = PGLeakyReLU(alpha=self.alpha)
        
    def forward(self, x):
        return self.act(self.conv(x))

    def extra_repr(self):
        return 'to_rgb {} to {}'.format(self.out_c, self.img_c)

    def save(self):
        torch.save(self, '{1}/weights/ToRGB_{0}x{0}'.format(self.spatial_resolution, getcwd()))
        return 'Generator block ToRGB_{0}x{0} is saved.\n'.format(self.spatial_resolution)


class PGfromRGB(Module):
    
    def __init__(self, out_channel, img_ch, alpha=0.2, spatial_resolution=4):
        super(PGfromRGB, self).__init__()
        
        self.out_c = out_channel
        self.in_c = img_ch
        self.alpha = alpha
        self.spatial_resolution = spatial_resolution
        
        self.conv = PGConv(self.in_c, self.out_c, 1, padding=0)
        self.act = PGLeakyReLU(alpha=self.alpha)
        
    def forward(self, x):
        return self.act(self.conv(x))

    def extra_repr(self):
        return 'from_RGB from {} to {}'.format(self.in_c, self.out_c)

    def save(self, directory=""):
        directory = getcwd() if directory == "" else directory
        torch.save(self, '{1}/weights/FromRGB_{0}x{0}'.format(self.spatial_resolution, directory))
        return 'Discriminator block FromRGB_{0}x{0} is saved.\n'.format(self.spatial_resolution)

a = _conv2d(10, 23, 3)