import torch
import torch.nn as nn
import torch.nn.functional as F

# These are the model classes for a DCGAN trained on CelebA.
# (Essentially the same as in https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

# They are meant to serve as a default GAN model to test other
# stuff (CutMix, etc.) until we have built up the U-Net GAN.

# Trained for a measly 5 epochs over celeba/train, but the samples
# look OK for what it's worth..

# I'll add 128x128 versions when I can reliably train a pair of them.

class DCGANGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        """Creates the generator.

        Note:
            Outputs 64x64 images, but easily extendable to 128x128.

        Args:
            nz: Dimentions of the sampled latent vector z.
            ngf: Scaling factor for the number of feature maps in the network.
            nc: Number of output channels (3, since we have RGB images here).
        """
        super(DCGANGenerator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 9
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, input):
        return self.main(input.view(input.shape[0], -1, 1, 1))

class DCGANDiscriminator(nn.Module):
    def __init__(self, ndf=64, nc=3):
        """Initialize the discriminator.
        Note:
            Expects (nc)x64x64 images, but easily extendable to 128x128.
        Args:
            ndf: Scaling factor for number of feature maps
            nc: Number of input channels (3, since we have RGB images)
        """
        super(DCGANDiscriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(input.shape[0], -1)


class UGANDiscriminator(nn.Module):
    """A miniature U-Net shaped discriminator.
    """
    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding, leak=0.2):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                        stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(leak, inplace=True)
        )

    def deconv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

            # state size. (ngf*8) x 8 x 8
    def __init__(self, ndf=64, nc=3):
        """Initialize the discriminator.
        Note:
            Expects (nc)x64x64 images, but easily extendable to 128x128.
        Args:
            ndf: Scaling factor for number of feature maps
            nc: Number of input channels (3, since we have RGB images)
        """
        super(UGANDiscriminator, self).__init__()
        ## The |_ part of the |__|-gan discriminator
        # input is (nc) x 64 x 64
        self.cblock1 = self.conv_block(nc, ndf, 4, 2, 1)
        # input is (ndf) x 32 x 32
        self.cblock2 = self.conv_block(ndf, ndf * 2, 4, 2, 1)
        # input is (ndf * 2) x 16 x 16
        self.cblock3 = self.conv_block(ndf * 2, ndf * 4, 4, 2, 1)
        # ... (ndf * 4) x 8 x 8
        self.cblock4 = self.conv_block(ndf * 4, ndf * 8, 4, 2, 1)

        self.out1 = nn.Conv2d(ndf * 8, 1, 4, 1, 0)

        ## Upconv time!
        self.dcblock1 = self.deconv_block(ndf * 8, ndf * 8, 4, 1, 0)
        self.dcblock2 = self.deconv_block(ndf * 8, ndf * 4, 4, 2, 1)
        self.dcblock3 = self.deconv_block(ndf * 4, ndf * 2, 4, 2, 1)
        self.dcblock4 = self.deconv_block(ndf * 2, ndf, 4, 2, 1)
        self.dcblock5 = self.deconv_block(ndf, 3, 4, 2, 1)

    def forward(self, x):
        # Need to store each activation
        act1 = self.cblock1(x)
        act2 = self.cblock2(act1)
        act3 = self.cblock3(act2)
        act4 = self.cblock4(act3)

        image_pred = torch.sigmoid(self.out1(act4)).view(act4.shape[0], -1)

        # Compute image real/fake prediction here

        # Compute upconvs (can reuse x here)
        x = F.relu(self.dcblock2(act4) + act3)
        x = F.relu(self.dcblock3(x) + act2)
        x = F.relu(self.dcblock4(x) + act1)
        x = self.dcblock5(x)

        per_pixel_preds = torch.sigmoid(x)
        return image_pred, per_pixel_preds

class DCGANGeneratorV2(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        """Creates the generator.

        Note:
            Outputs 128x128 images

        Args:
            nz: Dimentions of the sampled latent vector z.
            ngf: Scaling factor for the number of feature maps in the network.
            nc: Number of output channels (3, since we have RGB images here).
        """
        super(DCGANGeneratorV2, self).__init__()
        self.nz = nz
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 9
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input.view(input.shape[0], -1, 1, 1))
    def sample(self, device, num_samples=8):
        with torch.no_grad():
            zs = torch.randn(size=(num_samples, self.nz), device=device)
            return self.forward(zs)

class DCGANDiscriminatorV2(nn.Module):
    def __init__(self, ndf=64, nc=3):
        """Initialize the discriminator.
        Note:
            Expects (nc)x64x64 images, but easily extendable to 128x128.
        Args:
            ndf: Scaling factor for number of feature maps
            nc: Number of input channels (3, since we have RGB images)
        """
        super(DCGANDiscriminatorV2, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(input.shape[0], -1)

class UGANDiscriminatorV2(nn.Module):
    """A miniature U-Net shaped discriminator.
    """
    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding, leak=0.2):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                        stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(leak, inplace=True)
        )

    def deconv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

            # state size. (ngf*8) x 8 x 8
    def __init__(self, ndf=64, nc=3):
        """Initialize the discriminator.
        Note:
            Expects (nc)x64x64 images, but easily extendable to 128x128.
        Args:
            ndf: Scaling factor for number of feature maps
            nc: Number of input channels (3, since we have RGB images)
        """
        super(UGANDiscriminatorV2, self).__init__()
        ## The |_ part of the |__|-gan discriminator
        # input is (nc) x 64 x 64
        self.cblock1 = self.conv_block(nc, ndf, 4, 2, 1)
        # input is (ndf) x 32 x 32
        self.cblock2 = self.conv_block(ndf, ndf * 2, 4, 2, 1)
        # input is (ndf * 2) x 16 x 16
        self.cblock3 = self.conv_block(ndf * 2, ndf * 4, 4, 2, 1)
        # ... (ndf * 4) x 8 x 8
        self.cblock4 = self.conv_block(ndf * 4, ndf * 8, 4, 2, 1)

        self.cblock5 = self.conv_block(ndf * 8, ndf * 16, 4, 2, 1)

        self.out1 = nn.Conv2d(ndf * 16, 1, 4, 1, 0)

        ## Upconv time!
        self.dcblock1 = self.deconv_block(ndf * 16, ndf * 8, 4, 2, 1)
        self.dcblock2 = self.deconv_block(ndf * 8, ndf * 4, 4, 2, 1)
        self.dcblock3 = self.deconv_block(ndf * 4, ndf * 2, 4, 2, 1)
        self.dcblock4 = self.deconv_block(ndf * 2, ndf, 4, 2, 1)
        self.dcblock5 = self.deconv_block(ndf, 3, 4, 2, 1)

    def forward(self, x):
        # Need to store each activation
        act1 = self.cblock1(x)
        act2 = self.cblock2(act1)
        act3 = self.cblock3(act2)
        act4 = self.cblock4(act3)
        act5 = self.cblock5(act4)

        # Compute image real/fake prediction here
        image_pred = torch.sigmoid(self.out1(act5)).view(act5.shape[0], -1)

        # Compute upconvs (can reuse x here)
        x = F.relu(self.dcblock1(act5) + act4)
        x = F.relu(self.dcblock2(x) + act3)
        x = F.relu(self.dcblock3(x) + act2)
        x = F.relu(self.dcblock4(x) + act1)
        x = self.dcblock5(x)

        per_pixel_preds = torch.sigmoid(x)
        return image_pred, per_pixel_preds


