import torch
from torch.nn import functional as F

import numpy as np

from spade import SPADE
from misc.utils import downsample


class DenseFlowNetwork(torch.nn.Module):
    """
        Input (256, 256, 3)
        7x7 conv-32 Instance Normalization (256, 256, 32) ReLU
        3x3 conv-128 Instance Normalization (128, 128, 128) ReLU
        3x3 conv-512 Instance Normalization (64, 64, 512) ReLU

        SPADE Block (64,64,512)
        SPADE Block (64,64,512)
        SPADE Block (64,64,512)
        Pixel Shuffle (128,128,128)
        SPADE Block (128,128,128)
        Pixel Shuffle (256,256,32)

        7x7 Conv-2 Pixel Shuffle (256,256,2)
    """
    
    def __init__(self, num_channel=6, num_channel_modulation=3, hidden_size=512):
        super(DenseFlowNetwork, self ).__init__()

        # Convolutional Layers
        self.conv1 = torch.nn.Conv2d(num_channel, 32, kernel_size=7, stride=1, padding=3)
        self.conv1_bn = torch.nn.InstanceNorm2d(num_features=32, affine=True)
        self.conv1_relu = torch.nn.ReLU()
        
        self.conv2 = torch.nn.Conv2d(32, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_bn = torch.nn.InstanceNorm2d(num_features=128, affine=True)
        self.conv2_relu = torch.nn.ReLU()

        self.conv3 = torch.nn.Conv2d(128, 512, kernel_size=3, stride=2, padding=1)
        self.conv3_bn = torch.nn.InstanceNorm2d(num_features=512, affine=True)
        self.conv3_relu = torch.nn.ReLU()

        # SPADE Blocks
        self.spade_layer_1 = SPADE(512, num_channel_modulation, hidden_size)
        self.spade_layer_2 = SPADE(512, num_channel_modulation, hidden_size)
        self.spade_layer_3 = SPADE(512, num_channel_modulation, hidden_size)
        self.pixel_shuffle_1 = torch.nn.PixelShuffle(2)
        self.spade_layer_4 = SPADE(128, num_channel_modulation, hidden_size)
        self.pixel_shuffle_2 = torch.nn.PixelShuffle(2)

        # Final Convolutional Layer
        self.conv_4 = torch.nn.Conv2d(32, 2, kernel_size=7, stride=1, padding=3)

    def wrap(self, image, flow):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        flow: [2,h, w]
        """
        h, w = image.shape[-2], image.shape[-1]
        flow = downsample(flow, (h,w))
        grid_x, grid_y = np.mgrid[0:h, 0:w]
        grid = np.stack((grid_x, grid_y), axis=2)
        grid = torch.from_numpy(grid.astype(np.float32)).to(image.device)
        grid = grid.repeat([image.shape[0],1,1,1])
        flow = flow.permute(0,2,3,1)
        grid = grid - flow
        
        wrapped_image = F.grid_sample(image, grid)
        
        return wrapped_image

    def forward(self,  ref_image, ref_3d_face, style_3d_faces):

        # Concatenate the 3D face and the reference image
        concat = torch.cat((ref_image, ref_3d_face), dim=1)

        # Convolutional Layers
        h1 = self.conv1_relu(self.conv1_bn(self.conv1(concat)))
        h2 = self.conv2_relu(self.conv2_bn(self.conv2(h1)))
        h3 = self.conv3_relu(self.conv3_bn(self.conv3(h2)))
        

        # SPADE Blocks
        downsample_64 = downsample(style_3d_faces, (64,64))
        downsample_64 = downsample_64.view(downsample_64.size(0),-1, downsample_64.size(3), downsample_64.size(4) )

        spade_layer = self.spade_layer_1(h3, downsample_64)
        spade_layer = self.spade_layer_2(spade_layer, downsample_64)
        spade_layer = self.spade_layer_3(spade_layer, downsample_64)
        spade_layer = self.pixel_shuffle_1(spade_layer)
    
        downsample_128 = downsample(style_3d_faces, (128,128))        
        downsample_128 = downsample_128.view(downsample_128.size(0),-1, downsample_128.size(3), downsample_128.size(4) )

        spade_layer = self.spade_layer_4(spade_layer, downsample_128)
        spade_layer = self.pixel_shuffle_2(spade_layer)

        # Final Convolutional Layer
        conv_4 = self.conv_4(spade_layer)
        out = conv_4

        wrapped_h1 = self.wrap(h1, out)
        wrapped_h2 = self.wrap(h2, out)
        wrapped_h3 = self.wrap(h3, out)
        wrapped_ref = self.wrap(ref_image, out)

        return wrapped_h1, wrapped_h2, wrapped_h3, wrapped_ref

    
    
if __name__ == "__main__":
    # Test the Dense Flow Network
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dense_flow_network = DenseFlowNetwork(num_channel_modulation=9, hidden_size=512).to(device)
    print(dense_flow_network)
    input_image = torch.randn(2, 3, 256, 256).to(device)
    input_3d_face = torch.randn(2, 3, 256, 256).to(device)
    input_style_3d_face = torch.randn(2, 3, 3, 256, 256).to(device)
    output = dense_flow_network(input_image, input_3d_face, input_style_3d_face)
    print("h1: ", output[0].shape)
    print("h2: ", output[1].shape)
    print("h3: ", output[2].shape)
    print("out: ", output[3].shape)