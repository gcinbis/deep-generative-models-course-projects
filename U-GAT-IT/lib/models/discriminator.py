import torch.nn as nn
from lib.models.layers import ConvSNLeakyReLU, CAM, ConvSN2D


class LocalDiscriminator(nn.Module):
    def __init__(self, reflection=False):
        super().__init__()
        # ----------- Local Discriminator -----------
        # ----------- Encoder -----------
        # (Table-5: Part -> Encoder Down-sampling)
        # (h, w, 3) → (h/2, w/2, 64)
        # (h/2, w/2, 64) → (h/4, w/4, 128)
        # (h/4, w/4, 128) → (h/8, w/8, 256)
        # (h/8, w/8, 256) → (h/8 , w/8 , 512)
        self.encoder_downsampling = nn.Sequential(
            ConvSNLeakyReLU(3,    64, kernel_size=4, padding=1, stride=2, bias=True, reflection=reflection),
            ConvSNLeakyReLU(64,  128, kernel_size=4, padding=1, stride=2, bias=True, reflection=reflection),
            ConvSNLeakyReLU(128, 256, kernel_size=4, padding=1, stride=2, bias=True, reflection=reflection),
            ConvSNLeakyReLU(256, 512, kernel_size=4, padding=1, stride=1, bias=True, reflection=reflection)
        )

        # (Table-5: Part -> CAM of Discriminator)
        self.cam = CAM(512, act=nn.LeakyReLU(0.2), spectral_norm=True)

        # (Table-5: Part -> Classifier)
        # (h/8, w/8, 512) → (h/8, w/8, 1)
        self.classifier = ConvSN2D(512, 1, kernel_size=4, padding=1, stride=1, bias=False, reflection=reflection)

    def forward(self, x):
        x = self.encoder_downsampling(x)
        x, cam_logits, _ = self.cam(x)
        out = self.classifier(x)
        return out, cam_logits


class GlobalDiscriminator(nn.Module):
    def __init__(self, reflection=False):
        super().__init__()
        # ----------- Local Discriminator -----------
        # ----------- Encoder -----------
        # (Table-6: Part -> Encoder Down-sampling)
        # (h, w, 3) → (h/2, w/2, 64)
        # (h/2, w/2, 64) → (h/4, w/4, 128)
        # (h/4, w/4, 128) → (h/8, w/8, 256)
        # (h/8, w/8, 256) → (h/16 , w/16 , 512)
        # (h/16, w/16, 256) → (h/32 , w/32 , 1024)
        # (h/32 , w/32 , 1024) → (h/32 , w/32 , 2048)
        self.encoder_downsampling = nn.Sequential(
            ConvSNLeakyReLU(3,    64, kernel_size=4, padding=1, stride=2, bias=True, reflection=reflection),
            ConvSNLeakyReLU(64,  128, kernel_size=4, padding=1, stride=2, bias=True, reflection=reflection),
            ConvSNLeakyReLU(128, 256, kernel_size=4, padding=1, stride=2, bias=True, reflection=reflection),
            ConvSNLeakyReLU(256, 512, kernel_size=4, padding=1, stride=2, bias=True, reflection=reflection),
            ConvSNLeakyReLU(512, 1024, kernel_size=4, padding=1, stride=2, bias=True, reflection=reflection),
            ConvSNLeakyReLU(1024, 2048, kernel_size=4, padding=1, stride=1, bias=True, reflection=reflection),
        )

        # (Table-6: Part -> CAM of Discriminator)
        self.cam = CAM(2048, act=nn.LeakyReLU(0.2), spectral_norm=True)

        # (Table-6: Part -> Classifier)
        # (h/8, w/8, 512) → (h/8, w/8, 1)
        self.classifier = ConvSN2D(2048, 1, kernel_size=4, padding=1, stride=1, bias=False, reflection=reflection)

    def forward(self, x):
        x = self.encoder_downsampling(x)
        x, cam_logits, _ = self.cam(x)
        out = self.classifier(x)
        return out, cam_logits

