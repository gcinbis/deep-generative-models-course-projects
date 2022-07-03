import torch

"""
Implementation of the Spade (Spatially-Adaptive Normalization) from the paper:
https://arxiv.org/abs/1903.07291
"""

class SPADELayer(torch.nn.Module):
    """
        SPADE layer implementation
    """
    def __init__(self, num_channel= 3, num_channel_modulation=3, hidden_size= 256, kernel_size=3, stride=1, padding=1):
        super(SPADELayer, self).__init__()
        self.batch_norm = torch.nn.BatchNorm2d(num_channel)

        self.conv1 = torch.nn.Conv2d(num_channel_modulation, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.gamma = torch.nn.Conv2d(hidden_size, num_channel, kernel_size=kernel_size, stride=stride, padding=padding)  
        self.beta = torch.nn.Conv2d(hidden_size, num_channel, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x, modulation):

        x = self.batch_norm(x)

        conv_out  = self.conv1(modulation)

        gamma = self.gamma(conv_out)
        beta = self.beta(conv_out)

        return x + x * gamma + beta

class SPADE(torch.nn.Module):
    """
        SPADE block implementation
    """
    def __init__(self, num_channel=3, num_channel_modulation=3, hidden_size=256, kernel_size=3, stride=1, padding=1):
        super(SPADE, self).__init__()
        self.conv_1 = torch.nn.Conv2d(num_channel, num_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_2 = torch.nn.Conv2d(num_channel, num_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.spade_layer_1 = SPADELayer(num_channel, num_channel_modulation, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.spade_layer_2 = SPADELayer(num_channel, num_channel_modulation, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x, modulations):

        x = self.spade_layer_1(x, modulations)
        x = self.leaky_relu(x)
        x = self.conv_1(x)
        x = self.spade_layer_2(x, modulations)
        x = self.leaky_relu(x)
        x = self.conv_2(x)

        return x


if __name__ == "__main__":
    spade = SPADE(num_channel=512, num_channel_modulation=9) 
    x = torch.rand(1, 512, 64, 64)
    style = torch.rand(1, 9, 64, 64)
    normalized_content = spade(x, style)
    print(normalized_content.shape)

        