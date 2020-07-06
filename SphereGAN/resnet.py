from torch import nn
from torch.autograd import grad
import torch

from model_utils import *

class conv3x3(nn.Module):
    '''
        3x3 Conv utility class
    '''
    def __init__(self, input_dim, output_dim = None, bias = False):
        super(conv3x3, self).__init__()
        
        if output_dim == None:
            output_dim = input_dim

        self.conv = nn.Conv2d(input_dim, output_dim, 3, stride=1, padding=1, bias = bias)

    def forward(self, input):
        output = self.conv(input)
        return output


class ResidualBlockDownSample(nn.Module):
    '''
        Residual block with downsplampled shortcut, which reduces the spatial size by half.
        To be used in Discrimintor.

        Args:
            input_dim: number of features
            size: spatial dimension of the input.

    '''
    def __init__(self, input_dim, size=48):
        super(ResidualBlockDownSample, self).__init__()
        
        half_size = size//2

        self.avg_pool1 = nn.AdaptiveAvgPool2d((half_size,half_size))
        self.conv_shortcut = nn.Conv2d(input_dim, input_dim, kernel_size = 1)

        self.relu1 = nn.LeakyReLU(0.2)
        self.relu2 = nn.LeakyReLU(0.2)

        self.ln1 = nn.LayerNorm([input_dim, size, size])
        self.ln2 = nn.LayerNorm([input_dim, half_size, half_size])
        
        self.conv_1 = conv3x3(input_dim, input_dim, bias = False)
        self.conv_2 = conv3x3(input_dim, input_dim, bias = False)
        
        self.avg_pool2 = nn.AdaptiveAvgPool2d((half_size, half_size))
        
        
    def forward(self, input):
        shortcut = self.avg_pool1(input)
        shortcut = self.conv_shortcut(shortcut)

        output = self.ln1(input)
        output = self.relu1(output)
        output = self.conv_1(output)

        output = self.avg_pool2(output)

        output = self.ln2(output)
        output = self.relu2(output)
        output = self.conv_2(output)

        return shortcut + output

class ResidualBlockUpSample(nn.Module):
    """
        Residual block with upsampled shortcut, which doubles the spatial size.
        To be used in Discrimintor.

        Args:
            input_dim: number of features
            size: spatial dimension of the input.
    """
    def __init__(self, input_dim, size):
        super(ResidualBlockUpSample, self).__init__()
        
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.conv_shortcut = nn.Conv2d(input_dim, input_dim, kernel_size = 1)

        self.relu1 = nn.ReLU(True)
        self.relu2 = nn.ReLU(True)

        self.bn1 = nn.BatchNorm2d(input_dim)
        self.bn2 = nn.BatchNorm2d(input_dim)
        
        self.conv_1 = conv3x3(input_dim, input_dim, bias = False)
        self.conv_2 = conv3x3(input_dim, input_dim, bias = False)        
        self.upsample2 = nn.Upsample(scale_factor=2)
        
        
    def forward(self, input):
        shortcut = self.upsample1(input)
        shortcut = self.conv_shortcut(shortcut)

        output = self.bn1(input)
        output = self.relu1(output)
        output = self.conv_1(output)

        output = self.upsample2(output)

        output = self.bn2(output)
        output = self.relu2(output)
        output = self.conv_2(output)

        return shortcut + output

class ResidualBlock(nn.Module):
    """
        Standart residual block with layernorm instead of batchnorm.

    """
    def __init__(self, input_dim, size):
        super(ResidualBlock, self).__init__()
    
        self.conv_shortcut = nn.Conv2d(input_dim, input_dim, kernel_size = 1)

        self.relu1 = nn.LeakyReLU(0.2)
        self.relu2 = nn.LeakyReLU(0.2)

        self.ln1 = nn.LayerNorm([input_dim, size, size])
        self.ln2 = nn.LayerNorm([input_dim, size, size])
        
        self.conv_1 = conv3x3(input_dim, input_dim, bias = False)
        self.conv_2 = conv3x3(input_dim, input_dim, bias = False)        
        
        
    def forward(self, input):
        shortcut = self.conv_shortcut(input)

        output = self.ln1(input)
        output = self.relu1(output)
        output = self.conv_1(output)

        output = self.ln2(output)
        output = self.relu2(output)
        output = self.conv_2(output)

        return shortcut + output


class GeneratorResNet(nn.Module):
    """
        Resnet Generator Network

        Dimension flow:
            128 => dim,3,3 => dim,6,6 => dim,12,12 => dim,24,24 => dim,48,48 => 3,48,48

        Args:
            dim: number of features
    """
    def __init__(self, dim=256):
        super(GeneratorResNet, self).__init__()

        self.dim = dim
        
        self.ln1 = nn.Linear(128, 3*3*self.dim, bias=False)
        self.reshape = View((self.dim, 3, 3))
        self.bn_ln  = nn.BatchNorm2d(self.dim)
        
        self.rb1 = ResidualBlockUpSample(self.dim, size=3)
        self.rb2 = ResidualBlockUpSample(self.dim, size=6)
        self.rb3 = ResidualBlockUpSample(self.dim, size=12)
        self.rb4 = ResidualBlockUpSample(self.dim, size=24)
        self.bn  = nn.BatchNorm2d(self.dim)

        self.conv1 = conv3x3(self.dim, 3)
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = input
        
        output = self.ln1(output)
        output = self.reshape(output)
        output = self.bn_ln(output)
        
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output)
        output = self.bn(output)

        output = self.relu(output)
        output = self.conv1(output)
        output = self.tanh(output)

        return output

class DiscriminatorResNet(nn.Module):
    """
        Resnet Discriminator Network attached with Geometric block.

        Dimension flow:
            3,48,48 => dim,48,48 => dim,24,24 => dim,12,12 => dim,6,6 => dim,3,3

        Args:
            dim: number of features
    """
    def __init__(self, dim=256):
        super(DiscriminatorResNet, self).__init__()

        self.dim = dim

        self.conv1 = conv3x3(3, self.dim)
        self.rb1 = ResidualBlockDownSample(self.dim, size=48)
        self.rb2 = ResidualBlockDownSample(self.dim, size=24)
        self.rb3 = ResidualBlockDownSample(self.dim, size=12)
        self.rb4 = ResidualBlockDownSample(self.dim, size=6)
        self.rb5 = ResidualBlock(self.dim, size=3)

        self.gb = GeometricBlock(dim=self.dim, pool=True)


    def forward(self, input):
        output = input
        output = self.conv1(output)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output)
        output = self.rb5(output)
        output = self.gb(output)
        
        return output
