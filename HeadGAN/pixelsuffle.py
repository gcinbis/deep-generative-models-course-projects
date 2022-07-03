import torch

"""
Implementation of the PixelSuffleLayer from the paper:
https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shi_Real-Time_Single_Image_CVPR_2016_paper.pdf

"""

class PixelSuffleLayer(torch.nn.Module):

    def __init__(self, upscale_factor=2):

        super(PixelSuffleLayer, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        
        batch_size = x.size()[0]
        channel = x.size()[1]
        height = x.size()[2]
        width = x.size()[3]

        x = x.view(batch_size, channel // self.upscale_factor ** 2, self.upscale_factor, self.upscale_factor, height, width)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(batch_size, channel // self.upscale_factor ** 2, height * self.upscale_factor, width * self.upscale_factor)

        return x