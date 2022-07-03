from torch import nn


class Discriminator(nn.Module):
    """ Discriminator class for PD-GAN """

    def __init__(self):
        """
        Initialize the Discriminator for PDGAN model architecture.
        """
        super().__init__()

        "Convolution layers"
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

        "Layer Normalization Layers"
        self.ln1 = nn.LayerNorm([64, 64, 64])
        self.ln2 = nn.LayerNorm([64, 32, 32])
        self.ln3 = nn.LayerNorm([32, 16, 16])

        "ReLU Activations"
        self.lrelu1 = nn.LeakyReLU(0.1)
        self.lrelu2 = nn.LeakyReLU(0.1)
        self.lrelu3 = nn.LeakyReLU(0.1)
        self.lrelu4 = nn.LeakyReLU(0.1)

        "Sigmoid Activations"
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, img):
        """ Given an image real_or_fake_score"""

        out = self.conv1(img.clone().detach())
        out1 = self.lrelu1(out)

        out2 = self.conv2(out1)
        out2 = self.ln1(out2)
        out2 = self.lrelu2(out2)

        out3 = self.conv3(out2)
        out3 = self.ln2(out3)
        out3 = self.lrelu3(out3)

        out4 = self.conv4(out3)
        out4 = self.ln3(out4)
        out4 = self.lrelu4(out4)

        out5 = self.conv5(out4)

        out5 = self.sigmoid1(out5)

        return out5, [out1, out2, out3, out4, out5]
