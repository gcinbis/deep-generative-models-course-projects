import torch.utils.data
import torchvision.datasets
import torchvision.transforms as transforms
from torch.nn.functional import avg_pool2d
from collections import OrderedDict

class DataLoader:
    """
    This class provides functions to load images from datasets. 
    Available datasets are 'CIFAR10' and 'CelebA'.

    Args:
        dataset: name of the dataset
        batch_size: number of images in each batch
        num_blocks: number of blocks of generator and discriminator
        device: device that runs the model
        data_root: path containing the dataset. 
            if dataset is not already downloaded, 
            class downloads them automatically

    NOTE: IF YOU CAN NOT DOWNLOAD/USE CELEBA DATASET, CHECK README FILE. 
    """
    def __init__(self, dataset:str, batch_size=32, num_blocks=9, device="cuda", data_root="data/"):
        available_datasets = ["CIFAR10", "CelebA"]

        err = "invalid dataset. available datasets: " + ", ".join(available_datasets)
        assert dataset in available_datasets, err

        image_size = 4 * 2 ** (num_blocks - 1)
        self.num_blocks = num_blocks
        self.device = device

        if dataset=="CIFAR10":
            transform = transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Resize((image_size, image_size)),
                                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                          ])

            ds1 = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
            ds2 = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
            ds = torch.utils.data.ConcatDataset([ds1, ds2])
        elif dataset=="CelebA":
            transform = transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Resize((image_size, image_size)),
                                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                          ])

            ds = torchvision.datasets.CelebA(root=data_root, split='all', download=True, transform=transform)

        self.ds_len = len(ds)
        self.dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)


    def load_images(self):
        """
        This function returns a batch of images. 
        Since the model uses different sized versions of the same image,
        we resize and return a dictionary, whose batch with index i 
        corresponds to  i-th generator/discriminator block.
        """
        x = OrderedDict()
        imgs, _ = self.dataloader._get_iterator().__next__()
        imgs = imgs.to(self.device)
        
        for i in range(self.num_blocks-1, -1, -1):
            x[i] = imgs
            imgs = avg_pool2d(imgs, kernel_size=2)
        
        return x

    def get_len(self):
        """
        This function returns the size of the dataset.
        """
        return self.ds_len