import os
import torch.utils.data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from tqdm import tqdm
import pathlib
from flow_modules.misc import ShiftTransform, MnistGlowTransform
import torchvision.utils as vutils




def get_dataset(dataset_name, batch_size, data_root=None, train_workers=2, test_workers=2):
    assert dataset_name in ['cifar10', 'mnist', 'imagenet_32', 'imagenet_64'], "Invalid Dataset Name"

    if dataset_name == 'cifar10':
        if data_root is None:
            data_root = '../cifar_data'

        image_shape = [32, 32, 3]

        transform_train = transforms.Compose([
            ShiftTransform(3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))])

        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))])

        trainset = dsets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True,
                                                   num_workers=train_workers)

        testset = dsets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=True,
                                                  num_workers=test_workers)

    elif dataset_name == 'mnist':
        if data_root is None:
            data_root = '../mnist_data'

        image_shape = [32, 32, 3]

        transform_train = transforms.Compose([
            MnistGlowTransform(2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (1.0,))])

        transform_test = transforms.Compose([
            MnistGlowTransform(2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (1.0,))])

        trainset = dsets.MNIST(root=data_root, train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True,
                                                   num_workers=train_workers)

        testset = dsets.MNIST(root=data_root, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=True,
                                                  num_workers=train_workers)

    elif dataset_name == 'imagenet_32':
        if data_root is None:
            data_root = '../imagenet_data/'

        image_shape = [32, 32, 3]

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))])

        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))])

        trainset = dsets.ImageFolder(root=os.path.join(data_root, 'train_32x32'), transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True,
                                                   num_workers=train_workers)

        testset = dsets.ImageFolder(root=os.path.join(data_root, 'valid_32x32'), transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=True,
                                                  num_workers=test_workers)

    elif dataset_name == 'imagenet_64':
        if data_root is None:
            data_root = '../imagenet_data/'

        image_shape = [64, 64, 3]

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))])

        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))])

        trainset = dsets.ImageFolder(root=os.path.join(data_root, 'train_64x64'), transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True,
                                                   num_workers=train_workers)

        testset = dsets.ImageFolder(root=os.path.join(data_root, 'valid_64x64'), transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=True,
                                                  num_workers=test_workers)

    return train_loader, test_loader, image_shape


def rm_tree(pth):
    pth = pathlib.Path(pth)
    for child in pth.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()


def create_sample_directory(model, sample_dir, device='cpu', num_samples=10000, overwrite=False):
    """Creates a directory of randomly sampled images. Probably used to create
    a fake dataset to compute FID.
    Note:
        The FID authors suggest to use a minimum sample size of 10000 to not
        underestimate the FID value.
    Args:
        model:          The generator model
        sample_dir:     Target directory to create images
        num_samples:    Number of samples to create
        device:         CPU or CUDA device object
        overwrite:      If True, then overwrite sample directory if it already exists
    """
    sample_path = pathlib.Path(sample_dir)

    path_exists = sample_path.exists()
    if path_exists:
        if overwrite:
            # Careful!
            rm_tree(sample_path)
        else:
            raise FileExistsError('Directory already exists, not overwriting!')
    sample_path.mkdir(parents=True)
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(num_samples)):
            rev = model(None, None, reverse=True, eps_std=1.0)
            rev[torch.isnan(rev)] = -0.5
            rev = torch.clamp(rev, -0.5, 0.5)
            for j in range(0,16):
                vutils.save_image(rev[j].clone().detach().cpu(), str(sample_path / f'img{i}.png'), normalize=True)
                i+=1


def create_cifar10_sample_directory(cifar10_root_dir, sample_dir, num_samples=10000, overwrite=False,
                                   image_size=(32, 32)):
    """Selects and prepares randomly sampled CIFAR10 images.

    Note:
        The FID authors suggest to use a minimum sample size of 10000 to not
        underestimate the FID value.
    Args:
        celeba_root_dir:    Directory of the root 'celeba' directory
        sample_dir:         Target directory to create images
        num_samples:        Number of samples to create
        device:             CPU or CUDA device object
        overwrite:          If True, then overwrite sample directory if it already exists
        image_size:         Saved image size
    """
    sample_path = pathlib.Path(sample_dir)

    dataset = dsets.CIFAR10(root=cifar10_root_dir, download=False, transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [1, 1, 1])
    ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    dataloader_iter = iter(dataloader)

    for i in tqdm(range(num_samples)):
        try:
            sample = next(dataloader_iter)
        except StopIteration:
            # Re-roll the iterator
            dataloader_iter = iter(dataloader)
            sample = next(dataloader_iter)
        sample = sample[0].squeeze(0)
        # Unnormalize
        sample = 0.5 * sample + 0.5
        image = transforms.ToPILImage()(sample)
        image.save(str(sample_path / f'img{i}.png'), format='png')
