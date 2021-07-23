import torch
from torch import nn


def _conv_block(in_channel, out_channel, kernel_size, stride, padding):
    block = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channel),
        nn.LeakyReLU(0.2, True)  # TODO: check the negative slope?
    )

    return block


class ScaleAwareGen(nn.Module):
    def __init__(self, in_channel=3, filter_count=32, kernel_size=3, stride=1, padding=1,
                 num_block_phi=5, num_block_psi=4, device='cpu'):
        super().__init__()

        self.in_channel = in_channel
        self.filter_count = filter_count
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # first network
        self.num_block_phi = num_block_phi
        # second network
        self.num_block_psi = num_block_psi

        # device
        self.device = device

        self.phi_network = self._phi_network()
        self.psi_network = self._psi_network()

    def forward(self, x, y_prev):
        y = self.phi_network(x)
        # TODO: check dimensions!
        x_concat = torch.cat((x, y, y_prev), 1)

        attention_map = self.psi_network(x_concat)

        return attention_map * y + (1 - attention_map) * y_prev

    def _phi_network(self):
        phi_net = nn.Sequential()
        # in_channel = image channel, filter count
        phi_net.add_module('phi_0', _conv_block(self.in_channel, self.filter_count,
                                                self.kernel_size, self.stride, self.padding))

        for layer_idx in range(self.num_block_phi - 2):
            phi_net.add_module(f"phi_{layer_idx + 1}",
                               _conv_block(self.filter_count, self.filter_count,
                                           self.kernel_size, self.stride, self.padding))

        phi_net.add_module(f'phi_{self.num_block_phi-1}', nn.Conv2d(self.filter_count, self.in_channel,
                                              kernel_size=self.kernel_size, stride=self.stride, padding=self.padding))
        phi_net.add_module('phi_final_activation', nn.Tanh())

        return phi_net

    def _psi_network(self):
        psi_net = nn.Sequential()
        # in_channel = image channel * 3 since it takes Image, phi_out, Image^(n+1)
        psi_net.add_module('psi_0', _conv_block(3 * self.in_channel, self.filter_count,
                                                self.kernel_size, self.stride, self.padding))

        for layer_idx in range(self.num_block_psi - 2):
            psi_net.add_module(f"psi_{layer_idx + 1}",
                               _conv_block(self.filter_count, self.filter_count,
                                           self.kernel_size, self.stride, self.padding))

        psi_net.add_module(f'psi_{self.num_block_psi-1}', nn.Conv2d(self.filter_count, self.in_channel,
                                               kernel_size=self.kernel_size, stride=self.stride, padding=self.padding))
        psi_net.add_module('psi_final_activation', nn.Sigmoid())

        return psi_net


# PatchGAN with 11x11 Patch
class MarkovianDiscriminator(nn.Module):
    # should be same as Phi network
    def __init__(self, in_channel=3, filter_count=32, kernel_size=3, stride=1, padding=1, num_block=5):
        super().__init__()

        # in_channel = image channel, filter count
        self.net = nn.Sequential()
        self.net.add_module('discriminator_0', _conv_block(in_channel, filter_count, kernel_size, stride, padding))

        for layer_idx in range(num_block - 2):
            self.net.add_module(f"discriminator_{layer_idx + 1}",
                                _conv_block(filter_count, filter_count, kernel_size, stride, padding))

        self.net.add_module('discriminator_4', nn.Conv2d(filter_count, 1,
                                                         kernel_size=kernel_size, stride=stride, padding=padding))
        # TODO: add sigmoid or not?
        # self.net.add_module('last_layer,activation', nn.Sigmoid())

    def forward(self, x):
        y = self.net(x)
        return y
