from torchvision import datasets
from torchvision import transforms
import torch
from Parameters import hyper_parameters as parameters

trainset = datasets.MNIST(
    root="",
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.Resize(parameters["img_size"]), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]))

testset = datasets.MNIST(
    root="",
    train=False,
    download=True,
    transform=transforms.Compose(
        [transforms.Resize(parameters["img_size"]), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]))

trainloader = torch.utils.data.DataLoader(
    trainset,
    drop_last=True,
    batch_size=parameters["batch_size"],
    shuffle=True,
)

testloader = torch.utils.data.DataLoader(
    testset,
    drop_last=True,
    batch_size=parameters["batch_size"],
    shuffle=False,
)