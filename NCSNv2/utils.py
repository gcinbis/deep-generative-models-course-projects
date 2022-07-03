import os
import argparse
import torchvision.transforms as transforms
import numpy as np
import math
import torch
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

from PIL import Image


def parse_arguments():
    """
    Parse the arguments.
    :return: Arguments (argparse.Namespace).
    """
    parser = argparse.ArgumentParser(
        description='A reproduction attempt of Improved Techniques for Training Score-Based Generative Models.')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--dataset', choices=['CIFAR10', 'ToyDataset'], required=True)
    parser.add_argument('--model', choices=['NCSNv2', 'ToyModel'], required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=770)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--L', type=float, default=232)
    parser.add_argument('--sigma1', type=float, default=50)
    parser.add_argument('--eps', type=float, default=6.2e-6)
    parser.add_argument('--T', type=int, default=5)
    parser.add_argument('--reduction', choices=['sum', 'mean'], required=True)

    args = parser.parse_args()
    print('args:', args)

    return args


def get_name(args):
    """
    Build the name of the experiment from hyperparameters given as arguments.
    :param args: Arguments.
    :return: A string, the name of the experiment.
    """
    template = '{:s}_{:s}_{:s}_sigma{:d}_L{:d}_lr{:f}'

    name = template.format(
        args.dataset,
        args.model,
        args.reduction,
        args.sigma1,
        args.L,
        args.lr
    )

    return name


def sample(model, num_samples, noise_scales, eps, T, samples_folder, device):
    """
    Sample from the generative model using Annealed Langevin dynamics as described in 'Improved Techniques for Training
    Score-Based Generative Models' and 'Generative Modeling by Estimating Gradients of the Data Distribution'.
    :param model: A generative model.
    :param num_samples: The number of samples to be sampled.
    :param noise_scales: A numpy array of shape (L, ) containing each Gaussian noise scale. It is sorted in descending
        order.
    :param eps: The scaling constant used in step-size calculation.
    :param T: Number of iterations done at every noise scale.
    :param samples_folder: The path of the folder that the sampled images will be written in.
    :param device: Torch device (torch.device('cpu') or torch.device('cuda')).
    :return: A numpy array of shape (num_samples, 3, 32, 32). The values are clipped between 0 and 1.
    """
    print('Sampling from the model...')
    model = model.to(device)
    model.eval()
    num_batch = math.ceil(num_samples / 100)
    samples = []
    sigmaL = noise_scales[-1]
    with torch.no_grad():
        for i in tqdm(range(num_batch)):
            x = torch.rand(100, 3, 32, 32).to(device)
            for scale in noise_scales:
                step_size = eps * scale ** 2 / sigmaL ** 2
                for t in range(T):
                    z = torch.randn_like(x)
                    x = x + step_size * model(x) / scale + math.sqrt(2 * step_size) * z
            x = x + model(x) * sigmaL
            samples.append(x.detach().cpu())
    samples = torch.cat(samples, dim=0)
    samples = samples[:num_samples].numpy()

    samples = np.clip(samples, 0, 1)
    save_samples(samples, samples_folder)

    return samples


def calculate_noise_scales(sigma1, sigmaL, L):
    """
    Calculate Gaussian noise scales. The intermediate values are filled according to the geometric progression.
    :param sigma1: The largest noise scale.
    :param sigmaL: The lowest noise scale.
    :param L: The number of noise scales.
    :return: A numpy array of shape (L, ), containing the noise scale in descending order.
    """
    noise_scales = np.geomspace(sigma1, sigmaL, L).astype(np.float32)
    return noise_scales


def calculate_feature_mean_covar(samples, batch_size, device):
    """
    Given samples, features are extracted using Inception v3 network pretrained on ImageNet and their mean and
    covariance matrices are calculated.
    :param samples: A numpy array of shape (N, 3, 32, 32) where N is the number of samples.
    :param batch_size: Batch size to be used during feature calculation.
    :param device: Torch device (torch.device('cpu') or torch.device('cuda')).
    :return: A tuple containing two numpy arrays with shapes (2048, ) and (2048, 2048), respectively.
    """
    model = models.inception_v3(pretrained=True)
    model.dropout = nn.Identity()
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # Images are resized according to 'Rethinking the Inception Architecture for Computer Vision'
    transform = torch.nn.Sequential(
        normalize,
        transforms.Resize((299, 299))
    )
    features = []
    print('Extracting features...')
    with torch.no_grad():
        for batch_start in tqdm(range(0, samples.shape[0], batch_size)):
            images = samples[batch_start:min(batch_start + batch_size, samples.shape[0])]
            images = torch.tensor(images)
            images = transform(images)
            images = images.to(device)
            feature = model(images)
            feature = feature.detach().cpu()
            features.append(feature)
    features = torch.cat(features, dim=0)
    features = features.numpy()
    mu = np.mean(features, axis=0)
    sigma = np.cov(np.transpose(features))
    return mu, sigma


def calculate_fid(samples_mu, samples_sigma, real_mu, real_sigma):
    """
    Frechet Inception Distance is calculated given Inception v3 feature mean and covariance matrices of generated and
    real images.
    :param samples_mu: A numpy array of shape (2048, ). Mean matrix of the features of the generated images.
    :param samples_sigma: A numpy array of shape (2048, 2048). Covariance matrix of the features of the generated
        images.
    :param real_mu: A numpy array of shape (2048, ). Mean matrix of the features of the real images.
    :param real_sigma: A numpy array of shape (2048, 2048). Covariance matrix of the features of the real images.
    :return: A float, the calculated FID.
    """
    shur_decom = sqrtm(np.matmul(samples_sigma, real_sigma))
    fid = np.sum((samples_mu - real_mu) ** 2) + np.trace(samples_sigma + real_sigma - 2 * shur_decom)
    # fid2 = np.sum((samples_mu - real_mu)**2) + \
    #     np.trace(samples_sigma + real_sigma - 2 * np.linalg.cholesky(np.matmul(np.matmul(np.linalg.cholesky(samples_sigma), real_sigma), np.linalg.cholesky(samples_sigma))))

    fid = np.real(fid)

    return fid


def save_samples(samples, samples_folder):
    """
    Given sampled images, they are saved as png files to the given folder.
    :param samples: A numpy array of shape (N, 3, 32, 32) where N is the number of samples.
    :param samples_folder: A string, the path to the destination folder.
    """
    if not os.path.exists(samples_folder):
        os.makedirs(samples_folder)

    np.save(samples_folder + '.npy', samples)

    samples = np.moveaxis(samples, 1, -1)
    for i in range(samples.shape[0]):
        img = samples[i]
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)

        img.save(os.path.join(samples_folder, 'fig{:d}.png'.format(i)))


def draw_vector_field(model, dataset, device):
    """
    Visualizes the estimated scores as a vector field on 2D dataset.
    :param model: A generative model on 2D data.
    :param dataset: A torch.utils.data.Dataset object.
    :param device: Torch device (torch.device('cpu') or torch.device('cuda')).
    """
    num_instance = 20
    model.eval()
    x,y = np.meshgrid(np.linspace(-10,10,num_instance),np.linspace(-10,10,num_instance))
    x = np.expand_dims(x, axis=2)
    y = np.expand_dims(y, axis=2)
    inp = np.reshape(np.concatenate([x, y], axis=2), (-1, 2))
    inp = torch.from_numpy(inp).float().to(device)
    pred = model(inp)
    pred = pred.view(num_instance, num_instance, 2)
    pred = pred.detach().cpu().numpy()
    plt.plot(dataset.x[:, 0], dataset.x[:, 1], 'ko')
    plt.quiver(x, y, pred[:, :, 0], pred[:, :, 1])
    plt.show()


def remove_cached_real_feature_mu_sigmas():
    """Cleans the cached statistics of real images calculated for FID."""
    for filename in ['real_mu.npy', 'real_sigma.npy']:
        if os.path.exists(filename):
            os.remove(filename)