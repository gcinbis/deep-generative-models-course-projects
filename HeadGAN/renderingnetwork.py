import torch

from spade import SPADE
from adain import AdaIN
from pixelsuffle import PixelSuffleLayer
from misc.utils import downsample
class RenderingNetwork(torch.nn.Module):
    """
        Implementation the rendering network.
    """
    def __init__(self,num_channel=3):
        """
            Input (256,256,9)
            7x7 conv-32 Instance Norm  ReLU (256,256,32)
            3x3 conv-128 Instance Norm  ReLU (128,128,128)
            3x3 conv-512 Instance Norm  ReLU (512,256,512)
            SPADE
            AdaIN
            PixelSuffle
            SPADE
            AdaIN
            PixelSuffle
            SPADE
            AdaIN

            LeakyReLU
            7x7 conv-3 tanh (256,256,3)
        """
        super(RenderingNetwork, self).__init__()

        # Encoder
        self.conv1 = torch.nn.Conv2d(in_channels=9, out_channels=32, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv1_bn = torch.nn.InstanceNorm2d(num_features=32, affine=True)
        self.conv1_relu = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2_bn = torch.nn.InstanceNorm2d(num_features=128, affine=True)
        self.conv2_relu = torch.nn.ReLU()

        self.conv3 = torch.nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3_bn = torch.nn.InstanceNorm2d(num_features=512, affine=True)
        self.conv3_relu = torch.nn.ReLU()

        # Decoder
        self.spade_1 = SPADE(num_channel=512, num_channel_modulation=512)
        self.adain_1 = AdaIN(num_channel=512)
        self.pixel_suffle_1 = PixelSuffleLayer(upscale_factor=2)
        
        self.spade_2 = SPADE(num_channel=128, num_channel_modulation=128)
        self.adain_2 = AdaIN(num_channel=128)
        self.pixel_suffle_2 = PixelSuffleLayer(upscale_factor=2)
        
        self.spade_3 = SPADE(num_channel=32, num_channel_modulation=32)
        self.adain_3 = AdaIN(num_channel=32)
        self.spade_4 = SPADE(num_channel=32, num_channel_modulation=32)
                
        self.spade_3 = SPADE(num_channel=32, num_channel_modulation=32)
        self.adain_3 = AdaIN(num_channel=32)
        self.spade_4 = SPADE(num_channel=32, num_channel_modulation=3)

        # Final layer
        self.leaky_relu = torch.nn.LeakyReLU()
        self.conv_last = torch.nn.Conv2d(in_channels=32, out_channels=3, kernel_size=7, stride=1, padding=3, bias=False)
        self.tanh = torch.nn.Tanh()


    def forward(self, input_3d_face ,  wrapped_image, h1, h2, h3, audio_feature):
        
        x = input_3d_face.reshape(-1, 9, 256, 256)
        # Encoder
        x = self.conv1_relu(self.conv1_bn(self.conv1(x)))
        x = self.conv2_relu(self.conv2_bn(self.conv2(x)))
        x = self.conv3_relu(self.conv3_bn(self.conv3(x)))

        # Decoder
        x = self.spade_1(x, h3)
        x = self.adain_1(x, audio_feature)
        x = self.pixel_suffle_1(x)

        x = self.spade_2(x, h2)
        x = self.adain_2(x, audio_feature)
        x = self.pixel_suffle_2(x)

        x = self.spade_3(x, h1)
        x = self.adain_3(x, audio_feature)
        x = self.spade_4(x, wrapped_image)

        # Final layer
        
        x = self.leaky_relu(x)
        x = self.conv_last(x)
        x = self.tanh(x)
        
        return x

if __name__ == '__main__':
    # Test the network
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    renderer = RenderingNetwork().to(device)
    print(renderer)
    input_3d_face = torch.randn(2,9,256,256).to(device)
    wrapped_image = torch.randn(2,3,256,256).to(device)
    h3 = torch.randn(2,512,64,64).to(device)
    h2 = torch.randn(2,128,128,128).to(device)
    h1 = torch.randn(2,32,256,256).to(device)
    audio_feature = torch.randn(2,1,300).to(device)
    output = renderer(input_3d_face, wrapped_image, h1,h2,h3, audio_feature).cpu()
    print("Output: ", output.size())