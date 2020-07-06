import os

import torch
import torchvision
from torchvision import transforms


# weird hack to get element index
class SubsetWithIndices(torch.utils.data.Subset):
    def __getitem__(self, idx):
        return (*super().__getitem__(idx), idx)


def svhn_dataset(root, split):

    # dupgan paper supplies input image pixels in [-1 1] range for all channels
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    svhn_split = torchvision.datasets.SVHN(root=root, split=split, transform=transform, download=True)
    #remove digits >= 5
    svhn_split = torch.utils.data.Subset(svhn_split, [i for i in range(len(svhn_split)) if svhn_split.labels[i] <= 4])

    return svhn_split


def mnist_dataset(root, train):

    # dupgan paper supplies input image pixels in [-1 1] range for all channels
    # resize it to 32x32 make it 3 channel
    transform = transforms.Compose([transforms.Resize(32),  # transforms.Pad(2),
                                    transforms.Grayscale(3),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    mnist_set = torchvision.datasets.MNIST(root=root, train=train, transform=transform, download=True)
    #remove digits >= 5
    mnist_set = SubsetWithIndices(mnist_set, [i for i in range(len(mnist_set)) if mnist_set.targets[i] <= 4])

    return mnist_set


def generic_dataloader(device, dataset, shuffle, batch_size, pin_memory=True):

    if device == 'cuda':
        return torch.utils.data.DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, pin_memory=pin_memory)
    else:
        return torch.utils.data.DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)







