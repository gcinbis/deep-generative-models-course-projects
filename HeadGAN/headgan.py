import torch
import torchvision

from loss import headgan_loss

from denseflownetwork import DenseFlowNetwork
from renderingnetwork import RenderingNetwork
from discriminator import Discriminator

class HeadGAN(torch.nn.Module):
    def __init__(self):
        super(HeadGAN, self).__init__()

        # Dense flow Network
        self.F = DenseFlowNetwork(num_channel=6, num_channel_modulation=9, hidden_size=512)

        # Rendering Network
        self.R = RenderingNetwork(num_channel=9)

        # Discriminator Network
        self.D = Discriminator(num_channel=6)
        self.Dm = Discriminator(num_channel=6)
        
        self.vgg = torchvision.models.vgg19(pretrained=True)
        self.vgg.eval()



    def forward(self, ref_image, ref_3d_face, input_3d_faces, audio_features, gt):
        # Dense Flow Network
        h_1, h_2, h_3, y_ref = self.F(ref_image, ref_3d_face, input_3d_faces)
        
        # Rendering Network
        y_head_t = self.R(input_3d_faces, y_ref, h_1, h_2, h_3, audio_features)

        # Discriminator Network
        y_mount = y_head_t
        gt_mount = gt 

        x_t = input_3d_faces[:,0,:,:,:]

        losses = headgan_loss(x_t, audio_features, ref_image, gt, y_head_t, y_mount, gt_mount, self.D, self.Dm, self.vgg, lambda_l1=50, lambda_vgg=10, lambda_fm=10)

        
        return y_head_t, losses  
    

if __name__ == '__main__':
    # Test the HeadGAN
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    head_gan = HeadGAN().to(device)
    print(head_gan)
    bach_size = 16
    ref_image = torch.randn(bach_size, 3, 256, 256).to(device)
    ref_3d_face = torch.randn(bach_size, 3, 256, 256).to(device)
    input_3d_faces = torch.randn(bach_size, 3, 3, 256, 256).to(device)
    audio_features = torch.randn(bach_size, 1, 300).to(device)
    gt = torch.randn(bach_size, 3, 256, 256).to(device)
    out = head_gan(ref_image, ref_3d_face, input_3d_faces, audio_features, gt)
    print(out[0].shape)