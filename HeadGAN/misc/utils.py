"""
    Some utilities for the mics package.
"""
import torch
from torch.nn import functional as F

def downsample(x, size):
    """
    Downsample the image to specific size.
    """
    if len(x.size()) == 5:
        size = (x.size(2), size[0], size[1])
        return torch.nn.functional.interpolate(x, size=size, mode='nearest')
    return  torch.nn.functional.interpolate(x, size=size, mode='nearest')


def spatial_replication(feature, feature_like):
    """
    Replicate the feature map to specific size.
    """
    flatten = feature_like.view(feature_like.size(0), feature_like.size(1), -1)
    size = flatten.size(-1)
    factor = int(size / feature.size(-1))
    feature = feature.repeat(1, 1, factor)
    print(feature_like.size())
    print(feature.size())
    print(size - feature.shape[-1])
    padder = torch.nn.ConstantPad1d((0, size - feature.shape[-1]),0)
    feature = padder(feature)
    feature =  feature.view(feature_like.size(0), 1, feature_like.size(-2), feature_like.size(-1))
    return feature.expand_as(feature_like)
    
    
    

if __name__ == '__main__':
    image = torch.randn(2,3,256,256)
    audio = torch.randn(2,1,300)
    y = spatial_replication(audio, image)
    print(y.shape)