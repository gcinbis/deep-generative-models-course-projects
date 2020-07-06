import torch
import torch.nn as nn


def convBatchNormLeakyReLULayer(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding),
                         nn.BatchNorm2d(out_channels),
                         nn.LeakyReLU())


def trconvBatchNormLeakyReLULayer(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                            stride=stride, padding=padding),
                         nn.BatchNorm2d(out_channels),
                         nn.LeakyReLU())


def convReLUMaxPoolDropoutLayer(in_channels, out_channels, kernel_size, stride, padding, max_pool_size, dropout_prob):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding),
                         nn.ReLU(),
                         nn.MaxPool2d(kernel_size=max_pool_size),
                         nn.Dropout(dropout_prob))


class EncoderClassifier(nn.Module):

    # may test in_channels=3 (RGB) if UNIT approach doesn't work (RGBXY, second to last paragraph in UNIT paper apndx)
    def __init__(self, device, in_channels=3):
        super(EncoderClassifier, self).__init__()
        self.device = device

        # adapted from UNIT paper, table 4
        self.layer1 = convBatchNormLeakyReLULayer(in_channels, 64, 5, 2, 2)
        self.layer2 = convBatchNormLeakyReLULayer(64, 128, 5, 2, 2)
        self.layer3 = convBatchNormLeakyReLULayer(128, 256, 8, 1, 0)

        # UNIT paper has 1024 channels (neurons since inputs/outputs are 1x1) but those are mu,sigmas that
        # represent a 512x1 z. however dupgan does not do any sampling, so 512 channels suffice.
        # We also removed the extra fc layer that expands to 1024 nodes owing to this.
        # we also share all layers because dupgan does that

        self.latent = nn.Linear(256, 512)

        # Classifier C in DUPGAN paper picked as an fc+leakyrelu layer on top of latent representation
        # Paper does not specify, this is the simplest choice
        # not normalized!
        self.classifier = nn.Sequential(nn.LeakyReLU(), nn.Linear(512, 5))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = x.squeeze(dim=3)
        x = x.squeeze(dim=2)

        latent_out = self.latent(x)
        classifier_out = self.classifier(latent_out)

        return classifier_out, latent_out


class Generator(nn.Module):

    def __init__(self, device, out_channels=3):
        super(Generator, self).__init__()
        self.device = device

        # adapted from UNIT paper, table 4
        # extra element concatted to z in dupgan, representing whether z was from source/target
        # we halve the size of every layer and remove layer 5 since z is half the size of the z of UNIT
        # we also share all layers because dupgan does that

        # stride 2 is meaningless in a 1x1 image but that is what the unit paper did?
        self.layer1 = trconvBatchNormLeakyReLULayer(513, 256, 4, 2, 0)
        self.layer2 = trconvBatchNormLeakyReLULayer(256, 128, 4, 2, 1)
        self.layer3 = trconvBatchNormLeakyReLULayer(128, 64, 4, 2, 1)
        self.layer4 = nn.ConvTranspose2d(64, out_channels, 4, 2, 1)
        self.tanh = nn.Tanh()

    def forward(self, latent_out, domain_code):
        # domain code:
        # 0: want the svhn image
        # 1: want the mnist image

        # extra empty space later to be filled with domain code. allocated here for contigiousness in memory
        out = torch.empty((latent_out.shape[0], 513), dtype=latent_out.dtype, device=self.device)
        out[:, :-1] = latent_out
        out[:, -1] = domain_code
        #logging.info(out.shape)
        #logging.info(latent_out.shape)
        out = out.unsqueeze(dim=2)
        out = out.unsqueeze(dim=3)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.tanh(out)

        return out


class Discriminator(nn.Module):

    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        # adapted from UNIT paper, Table 6
        self.layer1 = convReLUMaxPoolDropoutLayer(in_channels, 64, 5, 1, 2, 2, 0.1)
        self.layer2 = convReLUMaxPoolDropoutLayer(64, 128, 5, 1, 2, 2, 0.1)
        self.layer3 = convReLUMaxPoolDropoutLayer(128, 256, 5, 1, 2, 2, 0.1)
        self.layer4 = convReLUMaxPoolDropoutLayer(256, 512, 5, 1, 2, 2, 0.1)

        # output is class+fake probs.
        # not normalized!
        self.discriminator = nn.Conv2d(512, 6, 2, 1, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        discriminator_out = self.discriminator(x)
        discriminator_out = discriminator_out.squeeze(dim=3)
        discriminator_out = discriminator_out.squeeze(dim=2)

        return discriminator_out














