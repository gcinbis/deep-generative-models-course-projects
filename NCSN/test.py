import os
import torch
import numpy as np

from tqdm import tqdm
from utils import calculate_feature_mean_covar, calculate_fid


def test(test_loader, samples, device, args):
    """
    Calculate FID score using test set for the model.
    :param model: The model with loaded parameters.
    :param test_loader: The DataLoader object of test dataset.
    :param device: Torch device (torch.device('cpu') or torch.device('cuda')).
    :param samples_folder: The folder to save the samples in.
    :param args: Arguments.
    :return: The calculated FID.
    """

    samples_mu, samples_sigma = calculate_feature_mean_covar(samples, args.batch_size, device)

    real_mu_path = 'real_mu.npy'
    real_sigma_path = 'real_sigma.npy'

    # If the mean and covariance matrices of real data are already calculated, it does not calculate them again.
    if not os.path.exists(real_mu_path):
        images = []
        print('Collecting CIFAR10 images...')
        for image, _ in tqdm(test_loader):
            images.append(image)
        images = torch.cat(images, dim=0)
        real_mu, real_sigma = calculate_feature_mean_covar(images, args.batch_size, device)
        np.save(real_mu_path, real_mu)
        np.save(real_sigma_path, real_sigma)

    real_mu = np.load(real_mu_path)
    real_sigma = np.load(real_sigma_path)

    fid = calculate_fid(samples_mu, samples_sigma, real_mu, real_sigma)
    return fid

