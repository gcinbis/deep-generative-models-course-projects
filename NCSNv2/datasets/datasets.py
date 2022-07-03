import torch
import torchvision
import torchvision.transforms as transforms


def get_dataloader(dataset_name, train_split, batch_size):
    """
    Get DataLoader instance of train/test dataset.
    :param dataset_name: The name of the dataset.
    :param train_split: If set to True, train loader is returned. Else Test loader is returned.
    :param batch_size: The batch size used in DataLoader.
    :return: A DataLoader instance.
    """
    if dataset_name == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=train_split,
            download=True,
            transform=transforms.ToTensor()
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=train_split,
            num_workers=2,
            pin_memory=True,
            drop_last=train_split
        )
    elif dataset_name == 'ToyDataset':
        from datasets.toy_dataset import ToyDataset
        dataset = ToyDataset('data/two_gaussians_inp.npy', 'data/two_gaussians_out.npy')
        dataloader = dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=train_split,
            num_workers=2,
            pin_memory=True,
            drop_last=train_split
        )
    else:
        raise ValueError('Unknown dataset:', dataset_name)
    return dataloader
