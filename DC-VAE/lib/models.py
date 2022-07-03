import torch
from torch import nn 
import torch.nn.functional as F


class ResConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, kernel, padding, pre_activation=False, downsample=False):
        super(ResConvBlock, self).__init__()
        self.pre_activation = pre_activation
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(ch_in, ch_out, kernel_size=kernel, padding=padding))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(ch_out, ch_out, kernel_size=kernel, padding=padding))
        self.shortcut = nn.utils.spectral_norm(nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0))
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.ds = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        #residual
        if self.pre_activation:
            hidden = self.relu(x)
        else:
            hidden = x
        hidden = self.relu(self.conv1(hidden))
        hidden = self.conv2(hidden)
        if self.downsample:
            hidden = self.ds(hidden)
            x = self.ds(x)
        sc = self.shortcut(x)
        return hidden + sc

class Encoder(nn.Module):
    def __init__(self, ch_in, hid_ch, z_dim):
        super(Encoder, self).__init__()
        self.block1 = ResConvBlock(ch_in=ch_in, ch_out=hid_ch, kernel=3, padding=1, pre_activation=False, downsample=True)
        self.block2 = ResConvBlock(ch_in=hid_ch, ch_out=hid_ch, kernel=3, padding=1, pre_activation=True, downsample=True)
        self.block3 = ResConvBlock(ch_in=hid_ch, ch_out=hid_ch, kernel=3, padding=1, pre_activation=True)
        self.block4 = ResConvBlock(ch_in=hid_ch, ch_out=hid_ch, kernel=3, padding=1, pre_activation=True)
        self.lin = nn.utils.spectral_norm(nn.Linear(hid_ch, 2*z_dim))
        self.relu = nn.ReLU()
    def forward(self, x):
        hidden = self.block1(x)
        hidden = self.block2(hidden)
        hidden = self.block3(hidden)
        hidden = self.relu(self.block4(hidden))
        hidden = hidden.sum(2).sum(2)
        out = self.lin(hidden)
        return out


class ConvBlock(nn.Module):
    ''' Pre-activation Conv Block with no Normalization '''

    def __init__(self, in_channel, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channels,kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.conv(F.relu(x))


class Decoder(nn.Module):
    def __init__(self, latent_dim, channel_dim):
        super().__init__()
        
        self.channel_dim = channel_dim
        self.latent_dim = latent_dim
    
        self.linear = nn.Linear(latent_dim, 4**2 * channel_dim)

        self.conv1 = ConvBlock(channel_dim, channel_dim)
        self.conv2 = ConvBlock(channel_dim, channel_dim)
        self.conv3 = ConvBlock(channel_dim, channel_dim)
        self.conv4 = ConvBlock(channel_dim, channel_dim)
        self.conv5 = ConvBlock(channel_dim, channel_dim)
        self.conv6 = ConvBlock(channel_dim, channel_dim)

        self.conv1x1_2 = nn.Conv2d(channel_dim, channel_dim, kernel_size=1)
        self.conv1x1_3 = nn.Conv2d(channel_dim, channel_dim, kernel_size=1)
        self.conv1x1_5 = nn.Conv2d(channel_dim, channel_dim, kernel_size=1)


        self.to_rgb = nn.Sequential(
            nn.BatchNorm2d(channel_dim),
            ConvBlock(channel_dim, 3),
            nn.Tanh()
        )

    def forward(self, x):
        
        x = self.linear(x).view(-1, self.channel_dim, 4, 4)
        skip1 = x

        # Cell 1
        x = nn.Upsample(scale_factor=2,mode='nearest')(x)
        x = self.conv1(x)
        skip2, skip3 = x, x
        x = self.conv2(x)
        x = x + nn.Upsample(scale_factor=2,mode='nearest')(skip1)

        # Cell 2
        skip4 = x
        x = nn.Upsample(scale_factor=2,mode='bilinear')(x)
        x = self.conv3(x)
        skip5 = x 
        x = x + self.conv1x1_2(nn.Upsample(scale_factor=2,mode='bilinear')(skip2))
        x = self.conv4(x)
        x = x + nn.Upsample(scale_factor=2,mode='bilinear')(skip4)

        # Cell 3
        x = nn.Upsample(scale_factor=2,mode='nearest')(x)
        x = self.conv5(x)
        x = x + self.conv1x1_3(nn.Upsample(scale_factor=4,mode='nearest')(skip3)) + self.conv1x1_5(nn.Upsample(scale_factor=2,mode='nearest')(skip5))
        x = self.conv6(x)

        out = self.to_rgb(x)
        return out


class Discriminator(nn.Module):
    def __init__(self, ch_in, hid_ch, cont_dim):
        super(Discriminator, self).__init__()
        self.block1 = ResConvBlock(ch_in=ch_in, ch_out=hid_ch, kernel=3, padding=1, pre_activation=False, downsample=True)
        self.block2 = ResConvBlock(ch_in=hid_ch, ch_out=hid_ch, kernel=3, padding=1, pre_activation=True, downsample=True)
        self.block3 = ResConvBlock(ch_in=hid_ch, ch_out=hid_ch, kernel=3, padding=1, pre_activation=True)
        self.block4 = ResConvBlock(ch_in=hid_ch, ch_out=hid_ch, kernel=3, padding=1, pre_activation=True)
        self.cont_conv = nn.Conv2d(hid_ch, 1, kernel_size=1, padding=1)
        self.cont_lin = nn.Linear(100, cont_dim)
        self.flatten = nn.Flatten()
        self.lin = nn.utils.spectral_norm(nn.Linear(hid_ch, cont_dim))
        self.disc_lin = nn.utils.spectral_norm(nn.Linear(cont_dim, 1))
        self.relu = nn.ReLU()
        
    def forward(self, x):
        hidden = self.block1(x)
        hidden = self.block2(hidden)
        cont_hidden = self.flatten(self.relu(self.cont_conv(hidden)))
        cont_hidden = self.relu(self.cont_lin(cont_hidden))
        #cont_out= self.cont_lin(cont_hidden)
        hidden = self.block3(hidden)
        hidden = self.relu(self.block4(hidden))
        hidden = hidden.sum(2).sum(2)
        hidden = self.lin(hidden)
        cont_out = cont_hidden + hidden
        disc_out = self.disc_lin(hidden)
        return disc_out, cont_out


class Model(nn.Module):
    def __init__(self, model_params):
        super().__init__()

        self.encoder = Encoder(**model_params['encoder'])
        self.decoder = Decoder(**model_params['decoder'])
        self.discriminator = Discriminator(**model_params['discriminator'])

    def forward(self, x):

        z_latent = self.encoder(x)
        z = self.reparametrize(z_latent)
        rec = self.decoder(z)
        return z_latent, rec
    
    def gen_from_noise(self, size):
        z_sampled = torch.randn(size).to(self.device)
        fake_data = self.decoder(z_sampled)
        return fake_data

    def reparametrize(self, z):

        log_var, mu = z.chunk(2, dim=1)
        N = torch.distributions.Normal(0, 1)
        eps = N.sample(mu.shape).to(self.device)
        std = torch.exp(log_var)
        z = eps*std + mu 

        return z



