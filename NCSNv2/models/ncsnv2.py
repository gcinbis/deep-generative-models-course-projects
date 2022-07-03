import torch
import torch.nn as nn


class InstanceNormPP(nn.Module):
    """The unconditional version of InstanceNorm++ operation defined in 'Generative Modeling by Estimating Gradients of
    the Data Distribution' and 'Improved Techniques for Training Score-Based Generative Models'."""
    def __init__(self, channel_size, epsilon=1e-8):
        """
        :param channel_size: The channel size of the input/output.
        :param epsilon: A small value added to denominator in the division for numeric stability.
        """
        super().__init__()
        self.gamma = nn.parameter.Parameter(torch.ones(channel_size, 1, 1))
        self.beta = nn.parameter.Parameter(torch.zeros(channel_size, 1, 1))
        self.alpha = nn.parameter.Parameter(torch.ones(channel_size, 1, 1))
        self.epsilon = epsilon

    def forward(self, x):
        mu = torch.mean(x, dim=(2, 3))
        var = torch.var(x, dim=(2, 3))
        sigma = torch.sqrt(var)

        m = torch.mean(mu, dim=1)
        v = torch.sqrt(torch.var(mu, dim=1))

        mu = mu.unsqueeze(2).unsqueeze(3)
        sigma = sigma.unsqueeze(2).unsqueeze(3)
        m = m.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        v = v.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        z = self.gamma * (x - mu) / (sigma + self.epsilon) + self.beta + self.alpha * (mu - m) / (v + self.epsilon)
        return z


class ResBlock(nn.Module):
    """Residual block with dilation."""
    def __init__(self, channel_size, dilation=1):
        """
        :param channel_size: The channel size through the block. (It remains the same.)
        :param dilation: Dilation parameter of convolution operation. As the down sampling is replaced with dilation,
        the dilation parameter has to be set in every ResBlock according to increase level.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channel_size, channel_size, kernel_size=3, padding=dilation, dilation=dilation)
        self.normalization1 = InstanceNormPP(channel_size)
        self.conv2 = nn.Conv2d(channel_size, channel_size, kernel_size=3, padding=dilation, dilation=dilation)
        self.normalization2 = InstanceNormPP(channel_size)

        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        inp = x
        x = self.conv1(x)
        x = self.normalization1(x)
        x = self.elu(x)
        x = self.conv2(x)
        x = self.normalization2(x)

        x = x + inp
        x = self.elu(x)
        return x


class ResBlockDown(nn.Module):
    """Residual Block for down sampling with dilation."""
    def __init__(self, inp_channel, out_channel, dilation=1):
        """
        :param inp_channel: The channel size of the input.
        :param out_channel: The channel size of the output.
        :param dilation: Dilation parameter of convolution operation. If the dilation is set to something greater than
            1, this block provides the increase in receptive field using dilation. If it is set to 1, the output will be
            down-sized using stride.
        """
        super().__init__()
        stride = 2 if dilation == 1 else 1
        self.conv1 = nn.Conv2d(inp_channel, out_channel, kernel_size=3, padding=dilation, stride=stride,
                               dilation=dilation)
        self.normalization1 = InstanceNormPP(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=dilation, dilation=dilation)
        self.normalization2 = InstanceNormPP(out_channel)

        self.elu = nn.ELU(inplace=True)

        self.inp_conv = nn.Conv2d(inp_channel, out_channel, kernel_size=1, stride=stride)
        self.inp_normalization = InstanceNormPP(out_channel)

    def forward(self, x):
        inp = x
        x = self.conv1(x)
        x = self.normalization1(x)
        x = self.elu(x)
        x = self.conv2(x)
        x = self.normalization2(x)

        inp = self.inp_conv(inp)
        inp = self.inp_normalization(inp)

        x = x + inp
        x = self.elu(x)
        return x


class ResNet(nn.Module):
    """ResNet-like network where batch normalizations are replaced with InstanceNorm++ and down sampling is replaced
    with dilation. For more detail, refer to 'RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic
    Segmentation' and 'Improved Techniques for Training Score-Based Generative Models'."""
    def __init__(self, channel_size):
        """
        :param channel_size: The initial channel size of ResNet. It is multiplied with 2 in later blocks.
        """
        super().__init__()
        self.conv_first = nn.Conv2d(3, channel_size, 3, padding=1)
        self.normalization_first = InstanceNormPP(channel_size)
        self.elu = nn.ELU(inplace=True)

        self.block1 = nn.Sequential(
            ResBlock(channel_size),
            ResBlock(channel_size)
        )

        self.block2 = nn.Sequential(
            ResBlockDown(channel_size, channel_size * 2),
            ResBlock(channel_size * 2, channel_size * 2)
        )

        self.block3 = nn.Sequential(
            ResBlockDown(channel_size * 2, channel_size * 2, dilation=2),
            ResBlock(channel_size * 2, dilation=2)
        )

        self.block4 = nn.Sequential(
            ResBlockDown(channel_size * 2, channel_size * 2, dilation=4),
            ResBlock(channel_size * 2, dilation=4)
        )

    def forward(self, x):
        x = self.conv_first(x)
        x = self.normalization_first(x)
        x = self.elu(x)

        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)

        return out1, out2, out3, out4


class ResidualConvUnit(nn.Module):
    """Residual Convolution Unit block defined in 'RefineNet: Multi-Path Refinement Networks for High-Resolution
    Semantic Segmentation'."""
    def __init__(self, channel_size):
        """
        :param channel_size: The channel size throughout the block.
        """
        super().__init__()
        self.elu1 = torch.nn.ELU(inplace=False)
        self.elu2 = torch.nn.ELU(inplace=True)
        self.conv1 = nn.Conv2d(channel_size, channel_size, 3, padding=1)
        self.conv2 = nn.Conv2d(channel_size, channel_size, 3, padding=1)

    def forward(self, inp):
        x = inp
        x = self.elu1(x)
        x = self.conv1(x)
        x = self.elu2(x)
        x = self.conv2(x)
        x = x + inp
        return x


class MultiResolutionFusion(nn.Module):
    """Multi Resolution Fusion block defined in 'RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic
    Segmentation'."""
    def __init__(self, inp_channel_1, inp_channel_2, upsample=False):
        """
        This branch merges the results of two branches. As the channel sizes of these blocks and the spatial
        dimensions may differ, the block adapts them accordingly.
        :param inp_channel_1: The channel size of the first input. This channel size has to be less than or equal to
            the second one.
        :param inp_channel_2: The channel size of the second input.
        :param upsample: The second input may be smaller than the first one in terms of spatial dimensionality. If this
            parameter is set to true, then the second input is upsampled accordingly.
        """
        super().__init__()
        assert inp_channel_1 <= inp_channel_2
        out_channel = inp_channel_1
        self.conv1 = nn.Conv2d(inp_channel_1, out_channel, 3, padding=1)
        self.conv2 = nn.Conv2d(inp_channel_2, out_channel, 3, padding=1)
        if upsample:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.upsample = None

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        if self.upsample:
            x2 = self.upsample(x2)
        return x1 + x2


class ChainedResidualPooling(nn.Module):
    """Chained Residual Pooling block defined in 'RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic
    Segmentation'."""
    def __init__(self, channel_size):
        """
        :param channel_size: The channel size throughout the block.
        """
        super().__init__()
        self.elu = nn.ELU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.conv1 = nn.Conv2d(channel_size, channel_size, 3, padding=1)
        self.conv2 = nn.Conv2d(channel_size, channel_size, 3, padding=1)

    def forward(self, x):
        x = self.elu(x)
        inp = x
        x = self.maxpool(x)
        x = self.conv1(x)
        inp = inp + x
        x = self.maxpool(x)
        x = self.conv2(x)
        inp = inp + x
        return inp


class RefineBlock(nn.Module):
    """RefineNet Block defined in 'RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation'.
    This block basically combines ResidualConvUnit, MultiResolutionFusion, and ChainedResidualPooling."""
    def __init__(self, channel_size_list, upsample=False):
        """
        :param channel_size_list: A list with one or two elements, each signifying the channel size of each input.
        :param upsample: If the spatial dimensions of inputs do not match, the second one can be upsampled accordingly
            if this parameter is set to True.
        """
        super().__init__()
        assert len(channel_size_list) <= 2
        min_channel_size = min(channel_size_list)
        self.rcu_units = []
        for i in range(len(channel_size_list)):
            channel_size = channel_size_list[i]
            rcu_unit = nn.Sequential(
                ResidualConvUnit(channel_size),
                ResidualConvUnit(channel_size)
            )
            self.rcu_units.append(rcu_unit)
        self.rcu_units = nn.ModuleList(self.rcu_units)

        if len(channel_size_list) > 1:
            self.mrf = MultiResolutionFusion(channel_size_list[0], channel_size_list[1], upsample=upsample)
        else:
            self.mrf = None

        self.crp = ChainedResidualPooling(min_channel_size)
        self.last_rcu = ResidualConvUnit(min_channel_size)

    def forward(self, x1, x2=None):
        x1 = self.rcu_units[0](x1)
        if self.mrf is not None:
            x2 = self.rcu_units[1](x2)
        if self.mrf is not None:
            x = self.mrf(x1, x2)
        else:
            x = x1
        x = self.crp(x)

        return x


class NCSNv2(nn.Module):
    """NCSNv2 model defined in 'Improved Techniques for Training Score-Based Generative Models'. This model is mainly
    based on RefineNet with some modifications described in the paper."""
    def __init__(self, channel_size):
        """
        :param channel_size: The initial channel size given to ResNet and used throughout the RefineBlock's accordingly.
        """
        super().__init__()
        self.resnet = ResNet(channel_size)
        self.rb4 = RefineBlock([channel_size * 2], upsample=False)
        self.rb3 = RefineBlock([channel_size * 2, channel_size * 2], upsample=False)
        self.rb2 = RefineBlock([channel_size * 2, channel_size * 2], upsample=False)
        self.rb1 = RefineBlock([channel_size, channel_size * 2], upsample=True)
        self.last_layer = nn.Sequential(
            ResidualConvUnit(channel_size),
            ResidualConvUnit(channel_size),
            nn.Conv2d(channel_size, 3, 3, padding=1)
        )

    def forward(self, x):
        out1, out2, out3, out4 = self.resnet(x)
        out4 = self.rb4(out4)
        out3 = self.rb3(out3, out4)
        out2 = self.rb2(out2, out3)
        out1 = self.rb1(out1, out2)
        x = self.last_layer(out1)
        return x
