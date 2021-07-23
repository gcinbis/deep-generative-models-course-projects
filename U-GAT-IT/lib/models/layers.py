import torch
import torch.nn as nn


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, bias=False, reflection=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if reflection:
            self.conv = nn.Sequential(
                nn.ReflectionPad2d(padding),
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=bias))

    def forward(self, x):
        return self.conv(x)


class ConvINReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, bias=False, reflection=False):
        super().__init__()
        self.conv = Conv2D(in_channels, out_channels, kernel_size, padding, stride, bias, reflection)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class ResINBlock(nn.Module):
    def __init__(self, in_channels, bias=False, reflection=False):
        super().__init__()

        self.block = nn.Sequential(
            ConvINReLU(in_channels, in_channels, 3, 1, 1, bias, reflection),
            Conv2D(in_channels, in_channels, 3, 1, 1, bias, reflection),
            nn.InstanceNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.block(x)


class CAM(nn.Module):
    def __init__(self, in_channels, act=nn.ReLU(), spectral_norm=False):
        super().__init__()
        self.mlp_gap = nn.Linear(in_channels, 1, bias=False)
        self.mlp_gmp = nn.Linear(in_channels, 1, bias=False)
        self.cam_conv = nn.Conv2d(in_channels*2, in_channels, kernel_size=1, stride=1, bias=True)
        self.cam_act = act

        if spectral_norm:
            self.mlp_gap = nn.utils.spectral_norm(self.mlp_gap)
            self.mlp_gmp = nn.utils.spectral_norm(self.mlp_gmp)

    def forward(self, x):
        B = x.shape[0]
        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1).view(B, -1)     # BxC
        gap_logits = self.mlp_gap(gap)
        gap_weight = list(self.mlp_gap.parameters())[0].unsqueeze(2).unsqueeze(3)
        gap = x * gap_weight    # BxCxHxW

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1).view(B, -1)
        gmp_logits = self.mlp_gmp(gmp)
        gmp_weight = list(self.mlp_gmp.parameters())[0].unsqueeze(2).unsqueeze(3)
        gmp = x * gmp_weight

        cam_logits = torch.cat([gap_logits, gmp_logits], dim=1)
        x = torch.cat([gap, gmp], dim=1)
        x = self.cam_act(self.cam_conv(x))
        heatmap = torch.sum(x, dim=1, keepdim=True)
        return x, cam_logits, heatmap


class AdaILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        # The value of ρ is initialized to 1 in the residual blocks of the decoder
        self.rho = nn.Parameter(torch.empty(1, num_features, 1, 1), requires_grad=True)
        self.rho.data.fill_(0.9)

    def forward(self, x, gamma, beta):
        B, C, _, _ = x.shape
        scale = gamma.unsqueeze(2).unsqueeze(3)
        bias = beta.unsqueeze(2).unsqueeze(3)

        in_mean = torch.mean(x.view(B, C, -1), dim=2).unsqueeze(2).unsqueeze(3)
        in_var = torch.var(x.view(B, C, -1), dim=2).unsqueeze(2).unsqueeze(3)
        a_in = (x - in_mean) / torch.sqrt((in_var + self.eps))

        ln_mean = torch.mean(x.view(B, -1), 1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        ln_var = torch.var(x.view(B, -1), 1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        a_ln = (x - ln_mean) / torch.sqrt((ln_var + self.eps))

        rho = self.rho.expand(B, -1, -1, -1)
        return scale * (rho * a_in + (1. - rho) * a_ln) + bias


class ILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        # The value of ρ is initialized to 0 in the up-sampling blocks of the decoder
        self.rho = nn.Parameter(torch.empty(1, num_features, 1, 1), requires_grad=True)
        self.gamma = nn.Parameter(torch.empty(1, num_features, 1, 1), requires_grad=True)
        self.beta = nn.Parameter(torch.empty(1, num_features, 1, 1), requires_grad=True)
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, x):
        B, C, _, _ = x.shape
        in_mean = torch.mean(x.view(B, C, -1), dim=2).unsqueeze(2).unsqueeze(3)
        in_var = torch.var(x.view(B, C, -1), dim=2).unsqueeze(2).unsqueeze(3)
        a_in = (x - in_mean) / torch.sqrt((in_var + self.eps))

        ln_mean = torch.mean(x.view(B, -1), 1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        ln_var = torch.var(x.view(B, -1), 1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        a_ln = (x - ln_mean) / torch.sqrt((ln_var + self.eps))

        rho = self.rho.expand(B, -1, -1, -1)
        scale = self.gamma.expand(B, -1, -1, -1)
        bias = self.gamma.expand(B, -1, -1, -1)
        return scale * (rho * a_in + (1. - rho) * a_ln) + bias


class ResAdaILNBlock(nn.Module):
    def __init__(self, in_channels, bias=False, reflection=False):
        super().__init__()
        self.conv1 = Conv2D(in_channels, in_channels, 3, 1, 1, bias=bias, reflection=reflection)
        self.norm1 = AdaILN(in_channels)

        self.conv2 = Conv2D(in_channels, in_channels, 3, 1, 1, bias=bias, reflection=reflection)
        self.norm2 = AdaILN(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x, gamma, beta):
        out = self.conv1(x)
        out = self.norm1(out, gamma, beta)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)
        return x + out


class UpConvILNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, scale_factor=2, bias=False, reflection=False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.conv = Conv2D(in_channels, out_channels, kernel_size, padding, stride=1, bias=bias, reflection=reflection)
        self.norm = ILN(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


# ------------------------ Discriminator Specific Layers ------------------------
class ConvSN2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, bias=False, reflection=False):
        super().__init__()
        if reflection:
            self.conv = nn.Sequential(
                nn.ReflectionPad2d(padding),
                nn.utils.spectral_norm(
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=bias)))
        else:
            self.conv = nn.utils.spectral_norm(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))

    def forward(self, x):
        return self.conv(x)


class ConvSNLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, bias=True, reflection=False):
        super().__init__()
        self.block = nn.Sequential(
            ConvSN2D(in_channels, out_channels, kernel_size, padding, stride, bias=bias, reflection=reflection),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.block(x)