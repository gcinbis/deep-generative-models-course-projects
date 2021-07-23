from abc import abstractmethod
import numpy as np

import torch
import torchvision.transforms as TF
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    MEAN = [0.5, 0.5, 0.5]
    STD = [0.5, 0.5, 0.5]

    def __init__(self, width, height, mean=None, std=None, mode="train"):
        assert mode in ["train", "test"], "Not available mode! Use one of ['train', 'test']"
        self.files = []
        self.width = width
        self.height = height
        self.mode = mode
        self.mean, self.std = mean, std
        if None in [mean, std]:
            self.mean = BaseDataset.MEAN
            self.std = BaseDataset.STD
        self.mean = torch.tensor(self.mean).float()
        self.std = torch.tensor(self.std).float()

        self.train_transform = TF.Compose([
            TF.RandomHorizontalFlip(),
            TF.Resize((self.height + 30, self.width + 30)),
            TF.RandomCrop((self.height, self.width)),
            TF.ToTensor(),
            TF.Normalize(self.mean, self.std)
        ])

        self.test_transform = TF.Compose([
            TF.Resize((self.height, self.width)),
            TF.ToTensor(),
            TF.Normalize(self.mean, self.std)
        ])

    @abstractmethod
    def _load_data_(self, index):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        fmt_str = f"Dataset: {self.__class__.__name__}\n"
        fmt_str += f"    # data: {self.__len__()}\n"
        return fmt_str

    def denormalize(self, tensors, inplace=True, device=None):
        _mean = torch.as_tensor(self.mean, dtype=torch.float, device=device)[None, :, None, None]
        _std = torch.as_tensor(self.std, dtype=torch.float, device=device)[None, :, None, None]
        if not inplace:
            tensors = tensors.clone()

        tensors.mul_(_std).add_(_mean)
        return tensors

    def normalize(self, tensors, inplace=True, device=None):
        _mean = torch.as_tensor(self.mean, dtype=torch.float, device=device)[None, :, None, None]
        _std = torch.as_tensor(self.std, dtype=torch.float, device=device)[None, :, None, None]
        if not inplace:
            tensors = tensors.clone()

        tensors.sub_(_mean).div_(_std)
        return tensors
