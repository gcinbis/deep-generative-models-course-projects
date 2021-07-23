from PIL import Image
from pathlib import Path

from lib.base import BaseDataset


class ImageFoldersAB(BaseDataset):
    def __init__(self, root, dataset_dir, width, height, mean, std, mode, **kwargs):
        super().__init__(width, height, mean, std, mode)
        self.root = Path(root).resolve()
        self.dataset_dir = self.root / dataset_dir
        self.files = self.get_files()
        self.transforms = self.train_transform if mode == 'train' else self.test_transform

    def get_files(self):
        domainA_imgs = list((self.dataset_dir / f'{self.mode}A').glob("*.*"))
        domainB_imgs = list((self.dataset_dir / f'{self.mode}B').glob("*.*"))
        return list(zip(domainA_imgs, domainB_imgs))

    def _load_data_(self, index):
        img_A_path, img_B_path = self.files[index][0], self.files[index][1]
        img_A = Image.open(img_A_path).convert("RGB")
        img_B = Image.open(img_B_path).convert('RGB')
        return img_A, img_B

    def __getitem__(self, index):
        img_A, img_B = self._load_data_(index)
        img_A = self.transforms(img_A)
        img_B = self.transforms(img_B)
        return {'A': img_A, 'B': img_B}


class ImageMemoryAB(BaseDataset):
    def __init__(self, root, dataset_dir, width, height, mean, std, mode, **kwargs):
        super().__init__(width, height, mean, std, mode)
        self.root = Path(root).resolve()
        self.dataset_dir = self.root / dataset_dir
        self.files = self.get_files()
        self.transforms = self.train_transform if mode == 'train' else self.test_transform
        self.dataset = self.load_all_images()

    def get_files(self):
        domainA_imgs = list((self.dataset_dir / f'{self.mode}A').glob("*.*"))
        domainB_imgs = list((self.dataset_dir / f'{self.mode}B').glob("*.*"))
        return list(zip(domainA_imgs, domainB_imgs))

    def _load_data_(self, index):
        img_A_path, img_B_path = self.files[index][0], self.files[index][1]
        img_A = Image.open(img_A_path).convert("RGB")
        img_B = Image.open(img_B_path).convert('RGB')
        return img_A, img_B

    def load_all_images(self):
        dataset = []
        for i in range(len(self.files)):
            dataset.append(self._load_data_(i))
        return dataset

    def __getitem__(self, index):
        img_A, img_B = self.dataset[index]
        img_A = self.transforms(img_A)
        img_B = self.transforms(img_B)
        return {'A': img_A, 'B': img_B}