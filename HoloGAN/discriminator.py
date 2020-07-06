"""
HoloGAN Discriminator implementation in PyTorch
May 17, 2020
"""
from torch import nn

class BasicBlock(nn.Module):
    """Basic Block defition of the Discriminator.
    """
    def __init__(self, in_planes, out_planes):
        super(BasicBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=2, padding=2)
        truncated_normal_initializer(self.conv2d.weight)
        nn.init.constant_(self.conv2d.bias, val=0.0)
        self.conv2d_spec_norm = nn.utils.spectral_norm(self.conv2d)
        self.instance_norm = nn.InstanceNorm2d(out_planes)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.conv2d_spec_norm(x)
        out = self.instance_norm(out)
        out = self.lrelu(out)
        return out

class Discriminator(nn.Module):
    """Discriminator of HoloGAN
    """
    def __init__(self, in_planes, out_planes, z_planes):
        super(Discriminator, self).__init__()
        self.conv2d = nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=2, padding=2)
        truncated_normal_initializer(self.conv2d.weight)
        nn.init.constant_(self.conv2d.bias, val=0.0)

        self.lrelu = nn.LeakyReLU(0.2)
        self.blocks = nn.Sequential(
            BasicBlock(out_planes*1, out_planes*2),
            BasicBlock(out_planes*2, out_planes*4),
            BasicBlock(out_planes*4, out_planes*8)
        )

        self.linear1 = nn.Linear(out_planes*8 * 4 * 4, 1)
        truncated_normal_initializer(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, val=0.0)

        self.linear2 = nn.Linear(out_planes*8 * 4 * 4, 128)
        truncated_normal_initializer(self.linear1.weight)
        nn.init.constant_(self.linear2.bias, val=0.0)

        self.linear3 = nn.Linear(128, z_planes)
        truncated_normal_initializer(self.linear1.weight)
        nn.init.constant_(self.linear3.bias, val=0.0)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        batch_size = x.size(0)
        x0 = self.lrelu(self.conv2d(x))
        x3 = self.blocks(x0)
        # Flatten
        x3 = x3.view(batch_size, -1)

        # Returning logits to determine whether the images are real or fake
        x4 = self.linear1(x3)

        # Recognition network for latent variables has an additional layer
        encoder = self.lrelu(self.linear2(x3))
        z_prediction = self.tanh(self.linear3(encoder))

        return x4, z_prediction

def truncated_normal_initializer(weight, mean=0, std=0.02):
    size = weight.shape
    tmp = weight.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    weight.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    weight.data.mul_(std).add_(mean)
