import numpy as np
import torch
import torchvision.models
from PIL import Image
from scipy import linalg
from torch import nn


# SIFID Implementation

class PartialInception(nn.Module):
    """This class loads and prepares the Inception model for SIFID calculations"""
    def __init__(self):
        super().__init__()
        self.partial_inception = self._load_partial_inception()

    def forward(self, x):
        x = 2 * x - 1
        y = self.partial_inception(x)
        return y

    @staticmethod
    def _load_partial_inception():
        # prepare inception model
        inception = torchvision.models.inception_v3(pretrained=True)
        inception.requires_grad_(False)
        inception.eval()

        # required inception blocks
        blocks = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
        ]

        return nn.Sequential(*blocks)

class SIFID:
    """SIFID calculator class"""
    def __init__(self, device='cuda'):
        self.device = device
        self.partial_inception = PartialInception().to(device)


    def calculate_sifid(self, inp1_path, inp2_path):
        """
        Calculates the sifid score between given two images
        :param inp1_path: image 1 path
        :param inp2_path: image 2 path
        :return:
        """
        img1 = self.read_normalize_img(inp1_path).to(self.device)
        img2 = self.read_normalize_img(inp2_path).to(self.device)

        img1_features = self._get_features(img1)
        img2_features = self._get_features(img2)

        fid = self._calculate_fid(img1_features, img2_features)

        return fid

    def _get_features(self, img):
        """
        Extracts necessary inception features for the given image.
        :param img: image
        :return: image features
        """
        img_features = self.partial_inception(img).detach().permute(0, 2, 3, 1).cpu().numpy()
        img_features = img_features.reshape(-1, img_features.shape[-1])

        return img_features

    def _calculate_fid(self, features1, features2):
        """
        Calculates the FID scores for the given features.
        FID = ||mu1 - mu2||^2 + trace(sigma1 + sigma2 - 2(sigma1 * sigma2)^(0.5))
        :param features1: image 1 features
        :param features2: image 2 fa
        :return:
        """
        mu1, sigma1 = self._get_stats(features1)
        mu2, sigma2 = self._get_stats(features2)

        diff = mu1 - mu2
        sqrt_sigma, _ = linalg.sqrtm(np.dot(sigma1, sigma2), disp=False)

        tr_covmean = np.trace(sqrt_sigma)

        return np.dot(diff, diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    @staticmethod
    def _get_stats(features):
        """Calculates statistics of the given features"""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)

        return mu, sigma

    @staticmethod
    def read_normalize_img(inp_path, method=Image.BICUBIC, resize_shape=(250, 250)):
        """Reads and c"""
        img = Image.open(inp_path).convert('RGB')
        if  img.size != resize_shape:
            img = img.resize(resize_shape, resample=method)

        img = np.array(img)/255
        img_tensor = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
        img_tensor = img_tensor.permute(0, 3, 1, 2)

        return img_tensor

if __name__ == '__main__':

    sifid = SIFID()
    val = sifid.calculate_sifid('data/horse_250_250.jpg', 'data/out_iter:4000_sc4_ABA.jpg')

    print(val)