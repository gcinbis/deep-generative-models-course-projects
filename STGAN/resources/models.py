"""
Details of the Network Structure can be found:
https://github.com/csmliu/STGAN/blob/8a206add4f2e87d4b13abaea9d6f77c964d46be9/paper/STGAN-suppl.pdf
"""

import torch
import torch.nn as nn

FC_DIM = 1024
N_LAYERS = 5
SHORTCUT_NUM = N_LAYERS - 1

# As the paper stated, 3x3 kernel size is used 
STU_KERNEL_SIZE = 3


class STU(nn.Module):

    def __init__(self, input_dim, output_dim, c, useNorm=True):
        super().__init__()

        self.c = c

        self.conv = nn.Conv2d(input_dim + output_dim, output_dim, STU_KERNEL_SIZE, 1, 1)
        self.upsample = nn.ConvTranspose2d(c + (2 * input_dim), output_dim, 4, 2, 1)

        self.reset = nn.Sequential(
            self.conv,
            nn.BatchNorm2d(output_dim),
            nn.Sigmoid()
        )

        self.update = nn.Sequential(
            self.conv,
            nn.BatchNorm2d(output_dim),
            nn.Sigmoid()
        )

        """
        Unlike GRU where ftl is adopted as the output of hidden state,
        they take sl as the output of hidden state and
        flt as the output of transformed encoder feature. 
        """

        self.intermediate = nn.Sequential(
            self.conv,
            nn.BatchNorm2d(output_dim),
            nn.Tanh()
        )

    def forward(self, input, state, att_diff):
        # Before upsample the state, we need to get initial shape to
        # scale the attribute difference
        n, _, h, w = state.shape

        # the difference attribute vector is stretched to have the same spatial size with the hidded state
        att_diff = att_diff.view((n, self.c, 1, 1))

        # If dimension=1, the expand function repeats the values in those dimensions
        # with specified number.
        # In this case:
        #   dimension 2 repeated h times
        #   dimension 3 repeated w times  
        att_diff = att_diff.expand((n, self.c, h, w))

        # Since feature maps across layers have different spatial size,
        # we upsample the hidden state s^(l+1) by using transposed convolution. 
        state = self.upsample(torch.cat([state, att_diff], dim=1))

        r = self.reset(torch.cat([input, state], dim=1))
        z = self.update(torch.cat([input, state], dim=1))

        output_state = r * state
        intermediate_output = self.intermediate(torch.cat([input, output_state], dim=1))
        output = ((1 - z) * state) + (z * intermediate_output)

        return output, output_state


class G_Enc(nn.Module):

    def __init__(self, dim=64, c=5):

        super().__init__()

        self.dim = dim
        self.c = c

        input_channels = 3
        output_channels = 2 * dim

        """
        Instead of Sequential form, ModuleList is used
        Since we need to access intermediate layers of the encoder
        in order to transfer them to the decoder.
        """
        self.enc = nn.ModuleList()

        for i in range(N_LAYERS):
            output_channels = dim * (2 ** i)

            self.enc.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, output_channels, 4, 2, 1),
                    nn.BatchNorm2d(output_channels),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                )
            )

            input_channels = output_channels

    def forward(self, x):

        outputs = []
        out = x

        for i in self.enc:
            out = i(out)
            outputs.append(out)

        return outputs


class G_Dec(nn.Module):

    def __init__(self, dim=64, c=5, stu_outputs=None):

        super().__init__()

        self.dim = dim
        self.c = c
        self.stu_outputs = stu_outputs

        """
        Instead of Sequential form, ModuleList is used
        Since we need to access intermediate layers of the decoder
        in order to concanate them with outputs of the STUs.
        """
        self.dec = nn.ModuleList()

        for i in range(N_LAYERS):

            # first layer
            if i == 0:

                input_channels = dim * (2 ** (N_LAYERS - 1)) + c
                output_channels = dim * (2 ** (N_LAYERS - 1))

                self.dec.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(input_channels, output_channels, 4, 2, 1),
                        nn.BatchNorm2d(output_channels),
                        nn.ReLU(inplace=True)
                    ))

            # last layer
            elif i == N_LAYERS - 1:

                # In the paper_supplementary, output dim stated as 3.
                self.dec.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(3 * dim, 3, 4, 2, 1),
                        nn.Tanh()
                    ))

            # For intermediate layers
            # Since all of them have shortcut layers, aka they are using STU
            # the channel calculation is done respectively.
            # Meaning that, for input channel, 3 * dim is used instead of 2 * dim 
            else:

                input_channels = (3 * dim) * (2 ** (N_LAYERS - 1 - i))
                output_channels = dim * (2 ** (N_LAYERS - 1 - i))

                self.dec.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(input_channels, output_channels, 4, 2, 1),
                        nn.BatchNorm2d(output_channels),
                        nn.ReLU(inplace=True)
                    ))

    def forward(self, latent, attribute, stu_outputs):

        out = torch.cat([latent, attribute], dim=1)

        out = self.dec[0](out)

        for i in range(1, N_LAYERS):
            out = torch.cat([out, stu_outputs[i - 1]], dim=1)
            out = self.dec[i](out)

        return out


class Generator(nn.Module):

    def __init__(self, dim=64, c=13):

        super().__init__()

        self.dim = dim
        self.c = c

        self.encoder = G_Enc(dim, c)
        self.decoder = G_Dec(dim, c)

        self.STUs = nn.ModuleList()

        for i in range(3, -1, -1):  # 3,2,1,0
            channel_num = dim * (2 ** i)
            self.STUs.append(STU(channel_num, channel_num, c, useNorm=False))

    def encode(self, x):
        return self.encoder.forward(x)

    def decode(self, latent, attribute, stu_outputs):
        return self.decoder.forward(latent, attribute, stu_outputs)

    def forward(self, image, attribute):

        encoder_outputs = self.encode(image)

        latent = encoder_outputs[-1]
        state = encoder_outputs[-1]

        n, _, h, w = latent.shape

        """
        This also can be done in the forward pass of the STU
        But this way we can loose the intuitivenes of STU
        """
        stu_outputs = []

        for i in range(1, N_LAYERS):
            stu_output, state = self.STUs[i - 1].forward(encoder_outputs[-(i + 1)], state, attribute)
            stu_outputs.append(stu_output)

        # If dimension=1, the expand function repeats the values in those dimensions
        # with specified number.
        # In this case:
        #   dimension 2 repeated h times
        #   dimension 3 repeated w times
        attribute = attribute.view((n, self.c, 1, 1)).expand((n, self.c, h, w))

        out = self.decoder(latent, attribute, stu_outputs)
        return out


class Discriminator(nn.Module):

    def __init__(self, x=384, dim=64, fc_dim=FC_DIM, c=5):
        """
        x = size of the image
        c = amount of attributes
        fc_dim = fully connected dimension. Its max is 1024
        """

        super().__init__()

        self.dim = dim
        self.fc_dim = fc_dim
        self.c = c

        input_channels = 3
        output_channels = 2 * dim

        stacked_conv_layers = []

        for i in range(N_LAYERS):
            output_channels = dim * (2 ** i)

            stacked_conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, output_channels, 4, 2, 1),
                    nn.InstanceNorm2d(output_channels),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                )
            )

            input_channels = output_channels

        self.stacked_conv = nn.Sequential(*stacked_conv_layers)

        feature_size = x // (2 ** N_LAYERS)

        fc_input_dim = dim * (2 ** (N_LAYERS - 1)) * (feature_size ** 2)

        self.D_adv = nn.Sequential(
            nn.Linear(fc_input_dim, fc_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(fc_dim, 1)
        )

        self.D_att = nn.Sequential(
            nn.Linear(fc_input_dim, fc_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(fc_dim, c),
        )

    def forward(self, x):
        intermediate = self.stacked_conv(x)

        intermediate = intermediate.view(intermediate.shape[0], -1)

        adv = self.D_adv(intermediate)
        att = self.D_att(intermediate)

        return adv, att
