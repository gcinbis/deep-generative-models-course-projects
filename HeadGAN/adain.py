import torch
"""
Implementation of the AdaIN(Adaptive Instance Normalisation) from the paper:
https://arxiv.org/abs/1703.06868
"""


def calculate_mean_std(x: torch.Tensor):
    """
    Calculates the mean and standard deviation of the input tensor.
    """
    batch_size = x.size()[0]
    channel_size = x.size()[1]

    variance = x.view(batch_size, channel_size, -1) +  1e-5
    variance = variance.var(dim=2)
    # Avoiding divide by zero error
    std = torch.sqrt(variance).view(batch_size, channel_size, 1, 1)
    mean = x.view(batch_size, channel_size, -1).mean(dim=2).view(batch_size, channel_size, 1, 1)

    return mean, std

def adaptive_instance_normalization(x, modulation):
    """
    Calculates the AdaIN(Adaptive Instance Normalisation) of the input tensor.
    """
    x_mean, x_std = calculate_mean_std(x)
    modulation_mean, modulation_std = calculate_mean_std(modulation)   

    modulation_mean = modulation_mean.mean(dim=1,keepdim=True)
    modulation_std = modulation_std.mean(dim=1,keepdim=True)

    normalized_content = (x - x_mean.expand_as(x)) / (x_std.expand_as(x))

    return normalized_content * modulation_std.expand_as(x) + modulation_mean.expand_as(x)


class AdaINLayer(torch.nn.Module):
    """
    Implementation of the AdanIN Layer implementation
    """
    def __init__(self):
        super(AdaINLayer, self).__init__()

    def forward(self, x, modulation):
        return adaptive_instance_normalization(x, modulation)

class AdaIN(torch.nn.Module):
    """
    Implementation of the AdanIN Block implementation
    """
    def __init__(self, num_channel=3, hidden_size=100, kernel_size=3, stride=1, padding=1):
        super(AdaIN, self).__init__()
        self.conv_1 = torch.nn.Conv2d(num_channel, num_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_2 = torch.nn.Conv2d(num_channel, num_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.adain_layer_1 = AdaINLayer()
        self.adain_layer_2 = AdaINLayer()


    def forward(self, x, modulation):

        x = self.adain_layer_1(x, modulation)
        x = self.leaky_relu(x)
        x = self.conv_1(x)
        x = self.adain_layer_2(x, modulation)
        x = self.leaky_relu(x)
        x = self.conv_2(x)

        return x

if __name__ == "__main__":
    content = torch.rand(2, 512, 64, 64)
    style = torch.zeros(2, 1, 300)
    adain = AdaIN(num_channel=512)
    normalized_content = adain(content, style)
    print(normalized_content.shape)