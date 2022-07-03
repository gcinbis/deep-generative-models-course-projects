from email.mime import audio
import torch
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

import copy 
"""
The Discriminator architecture is used in the paper following paper:
Semantic Image Synthesis with Spatially-Adaptive Normalization
https://arxiv.org/pdf/1903.07291.pdf
"""
class Discriminator(torch.nn.Module):
    def __init__(self, num_channel=6 ):
        """
            4x4 - Convolutional Layer -64, LReLU
            4x4 - Convolutional Layer -128, Spectral Normalization, LReLU
            4x4 - Convolutional Layer -256, Spectral Normalization, LReLU
            4x4 - Convolutional Layer -512, Spectral Normalization, LReLU
            4x4 - Convolutional Layer -1 
        """
        super(Discriminator, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_channel, 64, kernel_size=3, stride=2, padding=1)
        self.conv1_leaky_relu = torch.nn.LeakyReLU(0.2)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_bn = torch.nn.InstanceNorm2d(128)
        self.conv2_leaky_relu = torch.nn.LeakyReLU(0.2)
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv3_bn = torch.nn.InstanceNorm2d(256)
        self.conv3_leaky_relu = torch.nn.LeakyReLU(0.2)
        self.conv4 = torch.nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4_bn = torch.nn.InstanceNorm2d(512)
        self.conv4_leaky_relu = torch.nn.LeakyReLU(0.2)
        self.conv5 = torch.nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=0)
        self.flatten = torch.nn.Flatten()
        self.sigmoid = torch.nn.Sigmoid()
        self.fully_connected = torch.nn.Linear(196, 1)
        self.conv1 = spectral_norm(self.conv1)
        self.conv2 = spectral_norm(self.conv2)
        self.conv3 = spectral_norm(self.conv3)
        self.conv4 = spectral_norm(self.conv4)
        self.conv5 = spectral_norm(self.conv5)
        self.fully_connected = spectral_norm(self.fully_connected)

    def forward(self,  ground_truth, generated_image):
        concat = torch.cat((ground_truth, generated_image), dim=1)
        conv1 = self.conv1_leaky_relu(self.conv1(concat))
        conv2 = self.conv2_bn(self.conv2_leaky_relu(self.conv2(conv1)))
        conv3 = self.conv3_bn(self.conv3_leaky_relu(self.conv3(conv2)))
        conv4 = self.conv4_bn(self.conv4_leaky_relu(self.conv4(conv3)))
        conv5 = self.conv5(conv4)
        flatten = self.flatten(conv5)
        out_feature = conv5.view(conv5.size(0), 2, -1)
        fully_connected = self.fully_connected(flatten)
        out = self.sigmoid(fully_connected)
        return out, out_feature


if __name__ == "__main__":
    discriminator = Discriminator(num_channel=6)
    ground_truth = torch.rand(2, 3, 256, 256)
    generated_image = torch.rand(2, 3, 256, 256)
    audio_feature = torch.rand(2, 1, 256, 256)
    output = discriminator(ground_truth, generated_image)
    print(output[1].shape)