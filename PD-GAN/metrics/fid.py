import sys

sys.path.append('../library')

import torch
import pathlib
import numpy as np
from .inception import InceptionV3

from scipy import linalg
from library.dataset import Dataset
import argparse

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05' \
                  '-6726825d.pth '

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x


##################################################################
# This code adapted from  https://github.com/mseitzer/pytorch-fid #
##################################################################

def _compute_statistics_of_path(dataset, model, batch_size=50, dims=2048, device='cpu'):
    """

    :param dataset: dataset that contains images
    :param model: Inception V3 model
    :param batch_size: batch size of dataset
    :param dims: dimension of output
    :param device: training device cpu or cuda
    :return:
    """
    model.eval()

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=1)

    out_arr = np.empty((5000, dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            out = model(batch)[0]

        out = out.squeeze(3).squeeze(2).cpu().numpy()

        out_arr[start_idx:start_idx + out.shape[0]] = out

        start_idx = start_idx + out.shape[0]

    mu = np.mean(out_arr, axis=0)
    sigma = np.cov(out_arr, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    '''
    Calculate frechet distance
    :param mu1: mean of the first image
    :param sigma1: dev of first image
    :param mu2: mean of the second image
    :param sigma2: dev of second image
    :param eps: epsilon value
    :return:
    '''
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


def calculate_fid_given_paths(img_path, batch_size, gt_suffix, gen_img_suffix, device, dims=2048):
    """
    Calculates the FID between ground truth images and generated images
    :param img_path: Image path that contains generated and ground truth images
    :param batch_size: batch size of image
    :param gt_suffix: suffix of ground truth image
    :param gen_img_suffix: suffix of generated image
    :param device: training device cpu or cuda
    :param dims: dimension of output
    :return:
    """
    print("Calculating FID...")

    gt_dataset = Dataset(img_path, '*_ ' + gt_suffix + '.png')
    generated_dataset = Dataset(img_path, '*_' + gen_img_suffix + '.png')
    model = InceptionV3().to(device)
    m1, s1 = _compute_statistics_of_path(gt_dataset, model, batch_size,
                                         dims, device)
    m2, s2 = _compute_statistics_of_path(generated_dataset, model, batch_size,
                                         dims, device)

    return calculate_frechet_distance(m1, s1, m2, s2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--fid_images_path', default="../dataset/data_40_50/sky", type=str,
                        help='Image path that contains generated and ground truth images')
    parser.add_argument('--batch_size', default="100", type=int,
                        help='batch size of dataset')
    parser.add_argument('--gt_suffix', default="gt", type=str,
                        help='suffix of ground truth image')
    parser.add_argument('--gen_img_suffix', default="generated", type=str,
                        help='suffix of generated image')
    args = parser.parse_args()
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    fid_value = calculate_fid_given_paths(args.fid_images_path,
                                          args.batch_size, args.gt_suffix, args.gen_img_suffix,
                                          device)
    print('FID: ', fid_value)
