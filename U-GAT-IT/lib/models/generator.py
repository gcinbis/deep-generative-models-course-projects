import torch
import torch.nn as nn
from lib.models.layers import ConvINReLU, ResINBlock, CAM, ResAdaILNBlock, UpConvILNReLU, Conv2D


class Generator(nn.Module):
    def __init__(self, bias=False, reflection=False):
        super().__init__()
        # ----------- Encoder -----------
        # (Table-4: Part -> Encoder Down-sampling)
        # (h, w, 3) → (h, w, 64)
        # (h, w, 64) → ( h/2 , w/2 , 128)
        # (h/2 , w/2 , 128) → (h/4 , w/4 , 256)
        self.encoder_down = nn.Sequential(
            ConvINReLU(3, 64, kernel_size=7, padding=3, stride=1, bias=bias, reflection=reflection),
            ConvINReLU(64, 128, kernel_size=3, padding=1, stride=2, bias=bias, reflection=reflection),
            ConvINReLU(128, 256, kernel_size=3, padding=1, stride=2, bias=bias, reflection=reflection)
        )

        # (Table-4: Part -> Encoder Bottleneck)
        # (h/4 , w/4 , 256) → (h/4 , w/4 , 256) * 4 times
        self.encoder_bottleneck_1 = ResINBlock(256, bias=bias, reflection=reflection)
        self.encoder_bottleneck_2 = ResINBlock(256, bias=bias, reflection=reflection)
        self.encoder_bottleneck_3 = ResINBlock(256, bias=bias, reflection=reflection)
        self.encoder_bottleneck_4 = ResINBlock(256, bias=bias, reflection=reflection)

        # (Table-4: Part -> CAM of Generator)
        self.cam = CAM(256)

        # (Table-4: Part -> Gamma, Beta)
        self.gamma_mlp = nn.Sequential(
            nn.Linear(256, 256, bias=False), nn.ReLU(),     # (h/4 , w/4 , 256) → (1, 1, 256)
            nn.Linear(256, 256, bias=False), nn.ReLU(),     # (1, 1, 256) → (1, 1, 256)
            nn.Linear(256, 256, bias=False)                 # (1, 1, 256) → (1, 1, 256)
        )

        self.beta_mlp = nn.Sequential(
            nn.Linear(256, 256, bias=False), nn.ReLU(),     # (h/4 , w/4 , 256) → (1, 1, 256)
            nn.Linear(256, 256, bias=False), nn.ReLU(),     # (1, 1, 256) → (1, 1, 256)
            nn.Linear(256, 256, bias=False)                 # (1, 1, 256) → (1, 1, 256)
        )

        # ----------- Decoder -----------
        # (Table-4: Part -> Decoder Bottleneck)
        # (h/4 , w/4 , 256) → (h/4 , w/4 , 256) * 4 times
        self.decoder_bottleneck_1 = ResAdaILNBlock(256, bias=bias, reflection=reflection)
        self.decoder_bottleneck_2 = ResAdaILNBlock(256, bias=bias, reflection=reflection)
        self.decoder_bottleneck_3 = ResAdaILNBlock(256, bias=bias, reflection=reflection)
        self.decoder_bottleneck_4 = ResAdaILNBlock(256, bias=bias, reflection=reflection)

        # (Table-4: Part -> Decoder Up-sampling)
        # (h/4 , w/4 , 256) → (h/2 , w/2 , 128)
        # (h/2 , w/2 , 128) → (h , w , 64)
        # (h, w, 64) → (h, w, 3)
        self.decoder_up = nn.Sequential(
            UpConvILNReLU(256, 128, 3, 1, scale_factor=2, bias=bias, reflection=reflection),
            UpConvILNReLU(128,  64, 3, 1, scale_factor=2, bias=bias, reflection=reflection),
            Conv2D(64, 3, kernel_size=7, padding=3, stride=1, bias=bias, reflection=reflection),
            nn.Tanh()
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.encoder_down(x)
        x = self.encoder_bottleneck_1(x)
        x = self.encoder_bottleneck_2(x)
        x = self.encoder_bottleneck_3(x)
        x = self.encoder_bottleneck_4(x)
        x, cam_logits, heatmap = self.cam(x)

        x_avg = torch.nn.functional.adaptive_avg_pool2d(x, output_size=1).view(B, -1)
        gamma = self.gamma_mlp(x_avg)
        beta = self.beta_mlp(x_avg)

        x = self.decoder_bottleneck_1(x, gamma, beta)
        x = self.decoder_bottleneck_2(x, gamma, beta)
        x = self.decoder_bottleneck_3(x, gamma, beta)
        x = self.decoder_bottleneck_4(x, gamma, beta)

        out = self.decoder_up(x)
        if self.training:
            return out, cam_logits
        else:
            return out, heatmap

