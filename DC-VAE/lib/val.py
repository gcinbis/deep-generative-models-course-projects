import numpy as np
import torch
import torchvision
from .inception import InceptionV3
from tqdm import tqdm
import os
import glob
import time
from PIL import Image
from scipy.linalg import sqrtm
from pathlib import Path
import cv2
from shutil import rmtree

class PathData(torch.utils.data.Dataset):
    def __init__(self, data):
       self.data = data

    def __getitem__(self, i):
        data = self.data[i]
        img = Image.open(data).convert('RGB')
        transform=torchvision.transforms.ToTensor()
        return transform(img)

    def __len__(self):
        return len(self.data)

def calculate_mu_sigma(path, model, device, batch, dim):
    
    if os.path.isdir(path):

        data = sorted(glob.glob(os.path.join(path,'*.png')))
        dataset = PathData(data)
        generator = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=False, pin_memory=True, drop_last=True)

        outs = []
        model.eval()
        for img in tqdm(generator):

            img = img.to(device)

            with torch.no_grad():
                out = model(img)[0]

                out = out.squeeze(3).squeeze(2).cpu().numpy()

            outs.append(out)

        outs = np.array(outs)
        outs = outs.reshape(-1, dim)

        mu = np.mean(outs, axis=0)
        sig = np.cov(outs, rowvar=False)
        
    else:

        f = np.load(path)

        mu=f['mu']
        sig=f['sigma']
    
    return mu, sig

def calculate_fd(m1,s1,m2,s2):
   """ d^2 = ||mu_1 – mu_2||^2 + Tr(C_1 + C_2 – 2*sqrt(C_1*C_2)) """
   sqdiff = ((m1-m2)**2).sum()
   sqcov = sqrtm(s1.dot(s2))
   if np.iscomplexobj(sqcov):
       sqcov = sqcov.real
   fd = sqdiff + np.trace(s1 + s2 - 2 * sqcov) 
   return fd

def calculate_fid(path1, path2, device, dim=2048, batch=100):

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dim]
    inc_model = InceptionV3([block_idx]).to(device)

    m1,s1 = calculate_mu_sigma(path1, inc_model, device, batch, dim)
    m2,s2 = calculate_mu_sigma(path2, inc_model, device, batch, dim)
    
    return calculate_fd(m1,s1,m2,s2)

def eval(model, latent_dim, batch, device, loader):
    
    dirname = os.path.dirname(__file__)

    dir = os.path.join(dirname, f'../tmp/{time.time()}')

    Path(dir).mkdir(parents=True, exist_ok=True)

    cifar_gt_path = os.path.join(dirname, '../fid_stats_cifar10_train.npz')

    sample_num = 10000

    for idx in tqdm(range(0, sample_num, batch)):
        
        img_batch = model.gen_from_noise(size = (batch, latent_dim))

        #img_batch = (img_batch.permute(0,2,3,1).detach().cpu().numpy() * 127.5 + 127.5).astype(np.uint8)
        img_o = img_batch.mul_(127.5).add_(127.5).clamp(0.0,255.0)
        img_o = img_o.permute(0,2,3,1).to("cpu",torch.uint8).numpy()

        for ind, img in enumerate(img_o):
            cv2.imwrite(os.path.join(dir, f'{idx+ind}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    fid_samp = calculate_fid(cifar_gt_path, dir, device, batch = batch)
    rmtree(dir)

    dir = os.path.join(dirname, f'../tmp/{time.time()}')

    Path(dir).mkdir(parents=True, exist_ok=True)

    for idx, (img, _) in enumerate(tqdm(loader)):
        img = img.to(device)
        _, img_batch = model(img)

        #img_batch = (img_batch.permute(0,2,3,1).detach().cpu().numpy() * 127.5 + 127.5).astype(np.uint8)
        img_o = img_batch.mul_(127.5).add_(127.5).clamp(0.0,255.0)
        img_o = img_o.permute(0,2,3,1).to("cpu",torch.uint8).numpy()

        for ind, img in enumerate(img_o):
            cv2.imwrite(os.path.join(dir, f'{batch*idx+ind}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    fid_rec = calculate_fid(cifar_gt_path, dir, device, batch = batch)
    rmtree(dir)

    return fid_samp, fid_rec

