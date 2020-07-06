from models import DCGAN_64_Discriminator, DCGAN_64_Generator, StandardCNN_Discriminator, StandardCNN_Generator, InceptionV3
from torch.utils.data import Dataset as dst
from glob import glob
import torch
import torch.nn as nn
from torch.cuda import FloatTensor as Tensor
from torch import clamp
from torch.autograd import grad
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
import os
import sys
from PIL import Image
from argparse import ArgumentTypeError
import tarfile
import urllib
from scipy import linalg
try: 
    from tqdm import tqdm
except ImportError:
    def tqdm(x): return x

class Dataset(dst):

    def __init__(self, root, transforms, download):

        if(not os.path.exists(root) and download):

            print("Cat dataset is not found, it will be downloaded to datasets/cats_64. (24 MB)")

            cats_64_url = "https://drive.google.com/uc?id=19vLd3nuT3amuW4xlN7kTXoSYqWZlRRqx&export=download"
            urllib.request.urlretrieve(cats_64_url, os.path.join("datasets", "cats.tar.gz"))

            print("Cat dataset is downloaded, extracting...")
            cats_tar = tarfile.open(os.path.join("datasets", "cats.tar.gz"))
            cats_tar.extractall("datasets") 
            cats_tar.close()
            print("Extraction successful!")

        self.files = sorted(glob(root + '/*.png')) + sorted(glob(root + '/*.jpg'))

        self.transforms = transforms

    def __getitem__(self,index):

        return self.transforms(Image.open(self.files[index]))

    def __len__(self):

        return len(self.files)



def get_model(args):  
    """
    Returns the generator and discriminator models for the given model architecture and parameters such as no_BN, all_tanh and spec_norm.
        StandardCNN is the architecture described in the appendices I.1 of the paper, and DCGAN_64 is in appendices I.2.
        
    """
    #  

    if(args.model == "standard_cnn"):
        return (StandardCNN_Generator(no_BN = args.no_BN, all_tanh=args.all_tanh).to(args.device), 
                StandardCNN_Discriminator(no_BN = args.no_BN, all_tanh=args.all_tanh, spec_norm = args.spec_norm).to(args.device))

    if(args.model == "dcgan_64"):
        return (DCGAN_64_Generator(no_BN = args.no_BN, all_tanh=args.all_tanh).to(args.device), 
                DCGAN_64_Discriminator(no_BN = args.no_BN, all_tanh=args.all_tanh, spec_norm = args.spec_norm).to(args.device))


def get_loss(loss_type):
    """
    Returns the generator and discriminator losses for the given loss type.
        Relativistic generator losses use discriminator output for the real samples.
        Relativistic average losses use the average of discriminator outputs for both real and fake samples.
        Pre-calculated gradient penalty term is added to the discriminator losses using gradient penalty.
    """
     
    if(loss_type == "sgan"):

        loss = nn.BCEWithLogitsLoss()

        def disc_loss(C_real, C_fake):
            
            ones = torch.ones_like(C_fake)
            zeros = torch.zeros_like(C_fake)
            return (loss(C_real,ones) + loss(C_fake,zeros))

        def gen_loss(C_fake):
            
            ones = torch.ones_like(C_fake)
            return loss(C_fake,ones)

    elif(loss_type == "rsgan"):

        loss = nn.BCEWithLogitsLoss()

        def disc_loss(C_real, C_fake):
            
            ones = torch.ones_like(C_fake)
            return loss((C_real-C_fake),ones)

        def gen_loss(C_real, C_fake):
            
            ones = torch.ones_like(C_fake)
            return loss((C_fake-C_real),ones)
        

    elif(loss_type == "rasgan"):

        loss = nn.BCEWithLogitsLoss()

        def disc_loss(C_real, C_fake, C_avg_real, C_avg_fake):
            
            ones = torch.ones_like(C_fake)
            zeros = torch.zeros_like(C_fake)
            return (loss((C_real-C_avg_fake),ones) + loss((C_fake-C_avg_real),zeros))

        def gen_loss(C_real, C_fake, C_avg_real, C_avg_fake):
            
            ones = torch.ones_like(C_fake)
            zeros = torch.zeros_like(C_fake)
            return (loss((C_real-C_avg_fake),zeros) + loss((C_fake-C_avg_real),ones)) 

    elif(loss_type == "lsgan"):

        loss = nn.MSELoss()

        def disc_loss(C_real, C_fake):
            
            ones = torch.ones_like(C_fake)
            zeros = torch.zeros_like(C_fake)
            return (loss(C_real, zeros) + loss(C_fake, ones))

        def gen_loss(C_fake):
            
            zeros = torch.zeros_like(C_fake)
            return loss(C_fake,zeros)

    elif(loss_type == "ralsgan"):

        loss = nn.MSELoss()

        def disc_loss(C_real, C_fake, C_avg_real, C_avg_fake):
            
            ones = torch.ones_like(C_fake)
            return (loss((C_real-C_avg_fake), ones) + loss((C_fake-C_avg_real), -ones))

        def gen_loss(C_real, C_fake, C_avg_real, C_avg_fake):
            
            ones = torch.ones_like(C_fake)
            return (loss((C_fake-C_avg_real), ones) + loss((C_real-C_avg_fake),-ones)) 

    elif(loss_type == "hingegan"):

        def disc_loss(C_real, C_fake):

            ones = torch.ones_like(C_fake)
            return (clamp((ones-C_real), min=0).mean() + clamp((C_fake+ones), min=0).mean())

        def gen_loss(C_fake):
            
            return -C_fake.mean()

    elif(loss_type == "rahingegan"):

        def disc_loss(C_real, C_fake, C_avg_real, C_avg_fake):
            
            ones = torch.ones_like(C_fake)
            return (clamp((ones - C_real + C_avg_fake), min=0).mean() + clamp((ones + C_fake-C_avg_real), min=0).mean())

        def gen_loss(C_real, C_fake, C_avg_real, C_avg_fake):
            
            ones = torch.ones_like(C_fake)
            return (clamp((ones - C_fake + C_avg_real), min=0).mean() + clamp((ones + C_real - C_avg_fake), min=0).mean())

    elif(loss_type == "wgan-gp"):

        def disc_loss(C_real, C_fake, grad_pen):
            
            return (-C_real.mean() + C_fake.mean() + grad_pen)

        def gen_loss(C_fake):
            
            return -C_fake.mean()

    elif(loss_type == "rsgan-gp"):

        loss = nn.BCEWithLogitsLoss()

        def disc_loss(C_real, C_fake, grad_pen):
            
            ones = torch.ones_like(C_fake)
            return (loss((C_real - C_fake),ones) + grad_pen)

        def gen_loss(C_real, C_fake):
            
            ones = torch.ones_like(C_fake)
            return loss((C_fake-C_real), ones)

    elif(loss_type == "rasgan-gp"):

        loss = nn.BCEWithLogitsLoss()

        def disc_loss(C_real, C_fake, C_avg_real, C_avg_fake, grad_pen):
            
            ones = torch.ones_like(C_fake)
            zeros = torch.zeros_like(C_fake)
            return (loss((C_real-C_avg_fake),ones) + loss((C_fake-C_avg_real),zeros) + grad_pen)

        def gen_loss(C_real, C_fake, C_avg_real, C_avg_fake):
            
            ones = torch.ones_like(C_fake)
            zeros = torch.zeros_like(C_fake)
            return (loss((C_real-C_avg_fake),zeros) + loss((C_fake-C_avg_real),ones)) 


    return gen_loss,disc_loss

def grad_penalty(discriminator, x_hat, Lambda):
    """ 
    Calculates gradient penalty given the interpolated input x_hat.
    Lambda is the gradient penalty coefficient.
    """

    x_hat.requires_grad_(True)
    disc_out = discriminator(x_hat)
    grads = grad(outputs=disc_out, inputs=x_hat,
                 grad_outputs = torch.ones_like(disc_out),
                 create_graph=True)[0].view(x_hat.size(0),-1)

    return Lambda * torch.mean((grads.norm(p=2, dim=1) - 1)**2)



def get_dataset(dataset):
    """
    Returns the Dataset object of the given dataset, "cifar10" or "cat"
    
        For "cifar10", the class torchvision.datasets.CIFAR10 is used. It automatically downloads the dataset if it is not downloaded before.

        For "cat", the images in the folder "./datasets/cat_64" will be used for creating the dataset. If the folder does not exist, it will be automatically downloaded. 

    """
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if(dataset == "cifar10"):

        return CIFAR10(root='./datasets', train=True,
                                        download=True, transform=transform)

    if(dataset == "cat"):

        return Dataset(root=os.path.join("datasets", "cats_64"), transforms=transform, download=True)
    

def cycle(iterable, dset, device):
    """
    Restarts the dataloader iteration if it is iterated completely.

    Returns an iterable object of the batches which are sent to the preferred device(cpu or gpu).

        Returns x[0] for "cifar10" dataset since it returns a list of [images, labels] for its batches.

    """
    while True:
        for x in iterable:
            yield (x[0].to(device) if dset=='cifar10' else x.to(device))
            

def is_negative(value):
    """
    Checks if the given value as the argument is negative or not. If it is negative, give an error.

        Used for checking negative iteration frequency arguments.

    """

    if int(value) < 0:
        raise ArgumentTypeError(f"{value} should be non-negative")
    return int(value)


def sample_fid(generator, it, args, batch_size=100):
    """
    Generates samples to be used for calculating FID and saves them as a compressed numpy array.

        The number of samples to be generated is equal to the number of images in the training set (args.fid_sample).

    """

    generator.eval()

    with torch.no_grad():

        for i in range(0,args.fid_sample, batch_size):

            sys.stdout.write(f"\rGenerating {i}/{args.fid_sample}")

            if(args.fid_sample < batch_size+i):
                batch_size = args.fid_sample-i

            generated_samples = (generator(torch.randn(size=(batch_size,128,1,1), device=args.device))+1)*127.5 

            if(i == 0):
                arr = np.round_(generated_samples.cpu().permute(0,2,3,1).numpy()).astype(np.uint8)
            else:
                arr = np.concatenate((arr, np.round_(generated_samples.cpu().permute(0,2,3,1).numpy()).astype(np.uint8)), axis=0)
            
        np.savez_compressed(f"samples/{args.dataset}_{args.loss_type}_n_d_{args.d_iter}_b1_{args.beta1}_b2_{args.beta2}_b_size_{args.batch_size}_lr_{args.lr}_{it+1}" +  ( "_noBN" if args.no_BN else "") + ("_alltanh" if args.all_tanh else ""), images=arr)

    generator.train()


def extract_statistics(path, model, batch_size, use_cuda, verbose = False):
    """
        Computes and returns the mean and covariance matrix of the InceptionV3 features of the image dataset at the given path. 
        Arguments: 
            path: Dataset path. 
            model: The model for feature extraction (should be instance of models.InceptionV3)
            batch_size: Batch size to be used for feature extraction. 
            use_cuda: Boolean variable, use CUDA if True.
            verbose: Boolean variable, display progress if True.
        Return Values: 
            mu: Mean of the extracted features. 
            sigma: Covariance matrix of the extracted features. 
    """
    model.eval()
    
    # Load the data. 
    images = np.load(path)["images"]

    # Check for possible errors due to batch size \ number of samples
    if batch_size > len(images):
        print("Warning: Batch size larger than number of samples when computing FIDs!") 
        batch_size = len(images)
    
    remainder = len(images) % batch_size
        
    # Get the number of batches
    number_of_batches = len(images) // batch_size

    # Define the array of feature vectors
    features = np.empty(shape = (len(images), 2048))
    if verbose:
        print("Computing InceptionV3 activations...")

    for i in tqdm(range(number_of_batches)) if verbose else range(number_of_batches):
        
        # Get current batch
        batch = images[i * batch_size:(i + 1) * batch_size].astype(np.float32)

        # Reshape to (N, C, H, W)
        batch = batch.transpose(0, 3, 1, 2)

        # Scale down to [0, 1]
        batch /= 255

        # Convert batch to Tensor and load it to the selected device
        batch = torch.from_numpy(batch).type(torch.FloatTensor)

        if use_cuda:
            batch = batch.cuda()

        # Compute InceptionV3 features
        batch_of_features = model(batch)

        # Apply adaptive avg pooling to decrease number of features from 
        # 8 x 8 x 2048 to 1 x 1 x 2048 as per instructions from the original paper
        batch_of_features = nn.functional.adaptive_avg_pool2d(batch_of_features, output_size = (1,1))

        # Append batch of features to the feature list (after removing unnecessary dimensions).
        batch_of_features = batch_of_features.cpu().data.numpy().reshape(batch_size, 2048)
        features[i * batch_size:(i+1) * batch_size] = batch_of_features        

    # If the numnber of samples is not a multiple of batch size, handle remanining examples.
    if remainder != 0:
        i += 1 
        # reshape and rescale 
        batch = images[i * batch_size:].astype(np.float32)
        batch = batch.transpose(0, 3, 1, 2)
        batch /= 255
        
        # convert to tensor
        batch = torch.from_numpy(batch).type(torch.FloatTensor)
        if use_cuda:
            batch = batch.cuda()

        # process and save
        batch = nn.functional.adaptive_avg_pool2d(model(batch), output_size = (1,1))
        batch = batch.cpu().data.numpy().reshape(remainder, 2048)
        features[i * batch_size:] = batch

    if verbose: 
        print("InceptionV3 activations computed succesfully!")

    # Compute feature statistics
    mu = np.mean(features, axis = 0)
    sigma = np.cov(features, rowvar = False)

    return mu, sigma


def frechet_distance(mu1, mu2, sigma1, sigma2):
    """
        Computes and returns the frechet distance between the distributions represented by (mu1, sigma1) and (m2, sigma2). 
        Arguments: 
            mu1: The mean of distribution 1 (1-D numpy array).
            mu2: The mean of distribution 2 (1-D numpy array).
            sigma1: Covariance matrix of distribution 1 (2-D numpy array). 
            sigma2: Covariance matrix of distribution 2 (2-D numpy array). 
        Return Value: 
            frechet_distance: Frechet distance between the two distributions. 
                 
    """
    difference = mu1 - mu2
    difference_squared = difference.dot(difference)

    sqrt_of_prod, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    # Check for singularities
    if not np.isfinite(sqrt_of_prod).all():
        epsilon = np.eye(sigma1.shape[0]) * 1e-6
        sqrt_of_prod = linalg.sqrtm((sigma1 + epsilon).dot(sigma2 + epsilon))

    # Check for possible complex objects
    if np.iscomplexobj(sqrt_of_prod):
        sqrt_of_prod = sqrt_of_prod.real

    return (difference_squared + np.trace(sigma1) +
        np.trace(sigma2) - 2 * np.trace(sqrt_of_prod))


def calculate_fid(sample_path1, sample_path2, batch_size, use_cuda, verbose = True, model_path = "models/inception_v3.pt"):
    """
    Calculates the FID scores between the sets of samples saved in npz format under the given paths. The calculated scores
    will be saved to a file named fids.txt in the same directory.  
        Arguments: 
            sample_path1: Path for the first set of samples. 
            sample_path2: Path for the second set of samples. 
            batch_size: Batch size for FID computation, note that the number of samples should be a multiple of batch_size. 
            use_cuda: Boolean variable, cuda device will be used if set to True.
            verbose: Boolean variable, progress will be shown if set to True. 
            model_path: Path to a saved instance of models.InceptionV3, has default value: models/inception_v3.pt
        
    Note that FID between two sets of samples is symmetric, therefore sample_path1 and sample_path2 may be swapped without
    any effect on the resulting score. 
        
        Return Value: 
            fid_score: The (scalar) FID score between the samples in sample_path1 and sample_path2.
    """

    if (not os.path.exists(sample_path1)) or (not os.path.exists(sample_path2)):
        raise RuntimeError("Invalid path passed to calculate_fid!")
    
    if (not sample_path1.endswith(".npz")) or (not sample_path2.endswith(".npz")):
        raise RuntimeError("Invalid file type! Samples or statistics should be saved as .npz files.")
    
    # create a models folder if one does not exist
    if not os.path.exists("models/"): 
        os.mkdir("models")

    # check if inception_v3 is saved, if so just load
    if (os.path.exists(model_path)):
        inception_v3 = torch.load(model_path)
        print(f"InceptionV3 model is loaded from {model_path}.")
    
    # otherwise create a new object and save it
    else:
        print(f"InceptionV3 model will be created and saved to {model_path}.")
        inception_v3 = InceptionV3(verbose = verbose)
        torch.save(inception_v3, model_path)
        
    if use_cuda: 
        inception_v3.cuda()
        
    if(os.path.basename(os.path.normpath(sample_path1)).startswith("stats")):
        stat_file = np.load(sample_path1)
        mu1, sigma1 = stat_file['mu'], stat_file['sigma']
        
    else:
        mu1, sigma1 = extract_statistics(path = sample_path1,
                                        model = inception_v3, 
                                        batch_size = batch_size, 
                                        use_cuda = use_cuda,
                                        verbose = verbose)

    mu2, sigma2 = extract_statistics(path = sample_path2, 
                                    model = inception_v3, 
                                    batch_size = batch_size,
                                    use_cuda = use_cuda, 
                                    verbose = verbose)

    fid_score = frechet_distance(mu1, mu2, sigma1, sigma2)

    f = open("fids.txt", "a+")
    f.write(f"{os.path.basename(os.path.normpath(sample_path2))} FID {fid_score}\n")
    f.close()

    return fid_score
