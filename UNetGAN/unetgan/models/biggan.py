import torch
from torch._C import ModuleDict
import torch.nn as nn
import torch.nn.functional as F


### OTHER MODULES ###

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module

class UnConditionalBatchNorm(nn.Module):
    """
    Based on Conditional Batch Normalization
    https://arxiv.org/pdf/1707.00683.pdf

    It assumes Fully Connected embedding(float).
    Gain is increased by 1, in line with BigGAN.

    It is modified with self modularization:
    https://arxiv.org/pdf/1810.01365.pdf

    The batch-norm is changed to the PyTorch
    provided Spatial Batch Norm over channels.

    This is like StyleGAN projections.
    """
    def __init__(self, channels, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim
        self.channels = channels

        self.bn = nn.BatchNorm2d(channels)

        # In this work, we use a small one hidden layer feed-forward
        # network (MLP) with ReLU activation applied to the generator
        # input z.
        self.gain = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.channels)
        )
        self.bias = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.channels)
        )

        self.register_buffer('mu', torch.zeros(self.channels))
        self.register_buffer('sigma',  torch.ones(self.channels))

    def forward(self, x, condition):
        # Condition is latent vector

        g = self.gain(condition).view(-1, self.channels, 1, 1)
        b = self.bias(condition).view(-1, self.channels, 1, 1)

        out = self.bn(x)

        return out * (1 + g) + b

class SelfAttention(nn.Module):
    """
    Self-Attention Generative Adversarial Networks
    https://arxiv.org/pdf/1805.08318.pdf
    """
    def __init__(self, channel):
        super().__init__()

        self.channel = channel

        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.soft_argmax = nn.Softmax(dim=-1)

        self.f = spectral_norm(
            nn.Conv2d(channel, channel // 8, kernel_size=1, bias=False)
        )
        self.g = spectral_norm(
            nn.Conv2d(channel, channel // 8, kernel_size=1, bias=False)
        )
        self.h = spectral_norm(
            nn.Conv2d(channel, channel // 2, kernel_size=1, bias=False)
        )
        self.v = spectral_norm(
            nn.Conv2d(channel // 2, channel, kernel_size=1, bias=False)
        )

        # Quote: "γ is a learnable scalar and it is initialized as 0"
        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)
    
    """
    Exact flow from the paper.
    General formula: y_i = gamma*o_i + x_i
    """
    def forward(self, x):
        fx = self.f(x)
        gx = self.max_pool(self.g(x))
        hx = self.max_pool(self.h(x))

        h, w = x.shape[2:]

        fx = fx.view(-1, self.channel // 8, h * w).permute(0, 2, 1)
        gx = gx.view(-1, self.channel // 8, h * w // 4)

        s = torch.bmm(fx, gx)
        beta = self.soft_argmax(s).permute(0, 2, 1)

        hx = hx.view(-1, self.channel // 2, h * w // 4)
        o = self.v(torch.bmm(hx, beta).view(-1, self.channel // 2, h, w))

        return self.gamma * o + x


### GENERATOR ###

class GeneratorBlock(nn.Module):
    """
    BigGAN:
    https://arxiv.org/pdf/1809.11096.pdf
    Figure 15(b)
    """
    def __init__(self, in_c, out_c, latent_dim):
        super().__init__()

        self.relu = nn.ReLU()

        self.conv1 = spectral_norm(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        )
        self.conv2 = spectral_norm(
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        )

        self.conv_res = spectral_norm(
            nn.Conv2d(in_c, out_c, kernel_size=1)
        )

        self.norm1 = UnConditionalBatchNorm(in_c, latent_dim)
        self.norm2 = UnConditionalBatchNorm(out_c, latent_dim)

    def forward(self, x, y):
        
        a = self.relu(self.norm1(x, y))
        # Upsample, nearest is better.
        a = F.interpolate(a, scale_factor=2)
        a = self.conv1(a)
        a = self.relu(self.norm2(a, y))
        a = self.conv2(a)

        # Residual branch
        x = F.interpolate(x, scale_factor=2)
        x = self.conv_res(x)

        return a + x

class BigGANGenerator(nn.Module):
    """
    Spectral norms.
    No hierarchical latent space.
    """
    # If multiple device, calculate BN stats in G across all devices.
    def __init__(self, latent_dim=128, base_ch_width=64):
        super().__init__()

        self.latent_dim = latent_dim
        self.base_ch_width = base_ch_width
        
        # Channel multiplier(in-out): 16-16, 16-8, 8-4, 4-2, 2-1
        # Resolutions (square): 4 - 8 - 16 - 32 - 64 - 128
        self.initial_res = 4
        in_out_res = [
            [16, 16, 8  ],
            [16, 8 , 16 ],
            [8 , 4 , 32 ],
            [4 , 2 , 64 ],
            [2 , 1 , 128]
        ]
        for i in range(len(in_out_res)):
            in_out_res[i][0] *= base_ch_width
            in_out_res[i][1] *= base_ch_width

        self.latent_to_2d = spectral_norm(
            nn.Linear(latent_dim, in_out_res[0][0] * (self.initial_res ** 2))
        )

        self.module_list = nn.ModuleList([])
        for i, layer_info in enumerate(in_out_res):
            in_c, out_c, res = layer_info

            # Upsample at all layers
            self.module_list.append(
                GeneratorBlock(
                    in_c, out_c,
                    latent_dim=latent_dim
                )
            )

            # Attention at size 64
            if res == 64:
                self.attention_index = i
                self.attention = SelfAttention(out_c)

        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(in_out_res[-1][1]),
            nn.ReLU(),
            spectral_norm(
                nn.Conv2d(in_out_res[-1][1], 3, kernel_size=3, padding=1)
            )
        )

        # Output is an image with pixel values in [0, 1].
        self.tanh = nn.Tanh()
        
        self.initialize_weights()
        
    def forward(self, z):
        x = self.latent_to_2d(z)
        x = x.view(z.shape[0], -1, self.initial_res, self.initial_res)

        for i, module in enumerate(self.module_list):
            x = module(x, z)
            
            if i == self.attention_index:
                x = self.attention(x)

        x = self.output_layer(x)
        y = self.tanh(x)

        return y

    def initialize_weights(self):
        # Consider making this a global utility function.
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.Embedding)):
                nn.init.orthogonal_(module.weight)


### DISCRIMINATOR ###

class DiscriminatorBlock(nn.Module):
    """
    BigGAN:
    https://arxiv.org/pdf/1809.11096.pdf
    Figure 15(c)
    """
    def __init__(self, in_c, out_c):
        super().__init__()

        self.in_c = in_c
        self.out_c = out_c

        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(kernel_size=2)

        self.conv1 = spectral_norm(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        )
        self.conv2 = spectral_norm(
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        )

        self.conv_res = spectral_norm(
            nn.Conv2d(in_c, out_c, kernel_size=1)
        )

    def forward(self, x):
        # If not first layer, activate the residual data.
        a = self.relu(x) if self.in_c != 3 else x

        a = self.relu(self.conv1(a))
        a = self.conv2(a)
        # Downsample
        a = self.avg_pool(a)

        # Other branch
        b = self.conv_res(x)
        b = self.avg_pool(b)

        return a + b

class DiscriminatorUpBlock(nn.Module):
    """
    Nearly same as generator block.
    No normalization.
    """
    def __init__(self, in_c, out_c):
        super().__init__()

        self.relu = nn.ReLU()

        self.conv1 = spectral_norm(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        )
        self.conv2 = spectral_norm(
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        )

        self.conv_res = spectral_norm(
            nn.Conv2d(in_c, out_c, kernel_size=1)
        )
    
    def forward(self, x):
        
        a = self.relu(x)
        # Upsample, nearest is better.
        a = F.interpolate(a, scale_factor=2)
        a = self.conv1(a)
        a = self.relu(a)
        a = self.conv2(a)

        # Residual branch
        x = F.interpolate(x, scale_factor=2)
        x = self.conv_res(x)

        return a + x

class BigGANDiscriminator(nn.Module):
    def __init__(self, base_ch_width=64):
        super().__init__()

        self.base_ch_width = base_ch_width

        self.relu = nn.ReLU()

        self.down_module_list = nn.ModuleList([])
        self.up_module_list = nn.ModuleList([])

        # DOWN

        # First in_channel is plain 3, no extra width.
        # Channel multiplier(in-out): 3-1, 1-2, 2-4, 4-8, 8-16
        # Resolutions (square): 128 - 64 - 32 - 16 - 8 - 4
        in_out_res_down = [
            [3, 1 , 64],
            [1, 2 , 32],
            [2, 4 , 16],
            [4, 8 , 8 ],
            [8, 16, 4 ]
        ]
        for i in range(len(in_out_res_down)):
            if i != 0:
                in_out_res_down[i][0] *= base_ch_width
            in_out_res_down[i][1] *= base_ch_width

        for in_c, out_c, _ in in_out_res_down:
            # Downsample at all layers
            self.down_module_list.append(DiscriminatorBlock(in_c, out_c))

        # UP

        # First in_channel is plain 3, no extra width.
        # Channel multiplier(in-ou6t): 16-8, 16-4, 8-2, 4-1, 2-1
        # Resolutions (square): 4 - 8 - 16 - 32 - 64 - 128
        in_out_res_up = [
            [16, 8, 8  ],
            [16, 4, 16 ],
            [8 , 2, 32 ],
            [4 , 1, 64 ],
            [2 , 1, 128]
        ]
        for i in range(len(in_out_res_up)):
            in_out_res_up[i][0] *= base_ch_width
            in_out_res_up[i][1] *= base_ch_width

        for in_c, out_c, _ in in_out_res_up:
            # Upsample at all layers
            self.up_module_list.append(DiscriminatorUpBlock(in_c, out_c))

        # Pixel level classification
        self.output_layer = nn.Conv2d(base_ch_width, 1, kernel_size=1)

        # Classification layer
        self.classifier = spectral_norm(nn.Linear(base_ch_width * 4*4, 1))

    def forward(self, x):
        # Encoder
        d = x
        residuals = []
        for down_module in self.down_module_list:
            d = down_module(d)
            if down_module != self.down_module_list[-1]:
                residuals.append(d)

        # Classification result
        # Sum pooling
        df = torch.sum(self.relu(d), (2, 3))
        y = self.classifier(df)
        
        # Decoder
        u = d
        for i, up_module in enumerate(self.up_module_list):
            if i != 0:
                u = torch.cat((u, residuals[-i]), dim=1)
            u = up_module(u)

        pixel_y = self.output_layer(u)

        # Classifier and pixel level classifier, respectively.
        return y, pixel_y

    def initialize_weights(self):
        # Consider making this a global utility function.
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.Embedding)):
                nn.init.orthogonal_(module.weight)

