import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class InceptionV3(nn.Module):

    def __init__(self,):

        super(InceptionV3, self).__init__()

        inception = torch.load("./data/inception_v3.pt")

        self.blocks = nn.ModuleList()

        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        block1 = [
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block1))

        block2 = [
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
        ]
        self.blocks.append(nn.Sequential(*block2))

        block3 = [
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        ]
        self.blocks.append(nn.Sequential(*block3))

        self.upsample = nn.Upsample(size=(299, 299), mode='bilinear')

    def forward(self, input):
        output = self.upsample(input)

        output_list = []

        for block in self.blocks:
            output = block(output)
            output_list.append(output)

        return output
