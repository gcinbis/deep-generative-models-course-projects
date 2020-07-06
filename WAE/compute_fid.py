# -*- coding: utf-8 -*-
# --------------------------------------------------
#
# compute_FID.py
#
# Written by aliabbasi -*- ali.abbasi@metu.edu.tr
# Written by cetinsamet -*- cetin.samet@metu.edu.tr
# May, 2020
# --------------------------------------------------
from torch.nn.functional import adaptive_avg_pool2d 
import torch

from inception import InceptionV3
from random import sample 
from scipy import linalg
from PIL import Image 
import numpy as np
import pathlib 
import glob
import os

def imread(filename): 
    return np.asarray(Image.open(filename), dtype=np.uint8)[..., :3]


def get_activations(files, model, batch_size=50, dims=2048, cuda=False, verbose=False):
    """Calculates the activations of the pool_3 layer for all images """
    
    model.eval() 
    
    pred_arr = np.empty((files.shape[0], dims))

    # for i in range(0, len(files), batch_size):
    for i in range(0, files.shape[0], batch_size):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches), end='', flush=True)
        start = i
        end = i + batch_size 
        
        images = files[start:end, :,:,:]   

        batch = torch.from_numpy(images).type(torch.FloatTensor)
        if cuda:
            batch = batch.cuda()

        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling. 
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    if verbose:
        print(' done')

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """ Implementation of the Frechet Distance """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, model, batch_size=50, dims=2048, cuda=False, verbose=False):
    """ Calculation of the statistics used by the FID """
    act = get_activations(files, model, batch_size, dims, cuda, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def _compute_statistics_of_path(files, model, batch_size, dims, cuda):  
    m, s = calculate_activation_statistics(files, model, batch_size, dims, cuda)
    return m, s


def calculate_fid_given_samples(true_samples, generated_samples, batch_size=1, cuda='', dims=2048):
    """ Calculates the FID of two numpy array of samples """  
    
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()

    m1, s1 = _compute_statistics_of_path(true_samples, model, batch_size, dims, cuda)
    m2, s2 = _compute_statistics_of_path(generated_samples, model, batch_size, dims, cuda)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value 
    