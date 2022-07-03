import torch
from PIL import Image, ImageOps
import pathlib
from torchvision import transforms

# Mean and variance of normalization/unnormalization
# Here the same mean and var. values as values in Partial Convolution Encoder-Decoder Model
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]



class Dataset(torch.utils.data.Dataset):
    """
    Dataset class that used to get mask and images
    """

    def __init__(self, files, extension):
        """
        :param files: image or mask files
        :param extension: extension in image that specifies mask, ground truth or PC generated image
        :param is_mask_folder:
        """
        path_files = pathlib.Path(files)
        self.files = sorted([file for file in path_files.glob(extension)])
        self.extension = extension

    def __len__(self):
        """
        Get length of dataset
        :return:
        """
        return len(self.files)

    def __getitem__(self, i):
        """
        Get item in the dataset
        :param i: index value of the imahge
        :return:
        """
        path = self.files[i]
        # Load image or mask
        # Mask transformation is in Grayscale while image transformation is in RGB
        # Resize mask/image
        if self.extension != "*_mask.png":
            img_transform = transforms.Compose(
                [transforms.Resize(size=(256, 256)), transforms.ToTensor(),
                 transforms.Normalize(mean=MEAN, std=STD)])
            img = Image.open(path).convert('RGB')
            img = img_transform(img)
        else:
            mask_transform = transforms.Compose(
                [transforms.Resize(size=(256, 256)), transforms.ToTensor()])
            img = Image.open(path)
            img = ImageOps.grayscale(img)
            img = mask_transform(img)
        return img
