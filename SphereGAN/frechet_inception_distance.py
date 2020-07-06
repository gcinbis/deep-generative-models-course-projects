import os
import pathlib

import numpy as np
import torch
from scipy import linalg
from PIL import Image
import pathlib
from torch.nn.functional import adaptive_avg_pool2d
from torchvision.utils import save_image

import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

from inception import InceptionV3

def imread(filename):
    return np.asarray(Image.open(filename), dtype=np.uint8)[..., :3]


def calculate_activation_statistics(files, model, batch_size, device):
    model.eval()

    if batch_size > len(files):
        batch_size = len(files)

    pred_arr = np.empty((len(files), 2048))

    mean = np.array([0.485, 0.456, 0.406])[:, np.newaxis, np.newaxis]
    std = np.array([0.229, 0.224, 0.225])[:, np.newaxis, np.newaxis]

    for i in range(0, len(files), batch_size):
        if i%1024 == 0 :
            print(i, "/", len(files))
        start = i
        end = i + batch_size

        images = np.array([imread(str(f)).astype(np.float32) for f in files[start:end]])

        images = images.transpose((0, 3, 1, 2))
        images /= 255

        images -= mean
        images /= std

        batch = torch.from_numpy(images).type(torch.FloatTensor)
        
        batch = batch.to(device)

        pred = model(batch)
        pred_arr[start:end] = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    mu = np.mean(pred_arr, axis=0)
    sigma = np.cov(pred_arr, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def _compute_statistics_of_path(path, model, batch_size, device):

    path = pathlib.Path(path)
    files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
    m, s = calculate_activation_statistics(files, model, batch_size, device)
    return m, s


def calculate_fid_given_paths(paths, batch_size, device):
    
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    model = InceptionV3().to(device)

    m1, s1 = _compute_statistics_of_path(paths[0], model, batch_size, device)
    m2, s2 = _compute_statistics_of_path(paths[1], model, batch_size, device)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


