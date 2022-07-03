# implement evaluation functions
from multiprocessing import reduction
from matplotlib.image import imsave
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from data import AnimeFacesDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import histogram_feature_v2, truncation_trick, mixing_noise
from loss import hellinger_dist_loss
from model import HistoGAN, HistoGANAda
from torchvision.io import read_image
from torchvision.utils import save_image
import numpy as np
from torchvision.transforms import Resize

def random_interpolate_hists(batch_data, device="cpu"):
    B = batch_data.size(0)
    delta = torch.rand((B,1,1,1)).to(device)
    first_images = torch.index_select(batch_data, dim=0, index=torch.randint(0, B, (B,)).to(device))
    second_images = torch.index_select(batch_data, dim=0, index=torch.randint(0, B, (B,)).to(device))
    first_hist = histogram_feature_v2(first_images, device=device)
    second_hist = histogram_feature_v2(second_images, device=device)
    hist_t = delta*first_hist + (1-delta)*second_hist
    return hist_t

# Device "cpu" is advised by torch metrics
def fid_scores(generator, dataloader, fid_batch=8, num_gen_layers=5, latent_dim=512, mixing_prob=0.9,  device="cpu"):

    fid = FrechetInceptionDistance(feature=2048, reset_real_features=True)
    
    fids = []
    num_generated = 0
    for batch_data in dataloader:
        batch_data = batch_data.to(device)
        z = mixing_noise(fid_batch, num_gen_layers, latent_dim, mixing_prob).to(device) 
        target_hist = random_interpolate_hists(batch_data)
        fake_data, _ = generator(z, target_hist, test=True)
        batch_data = batch_data*255
        fake_data = fake_data*255
        fake_data = fake_data.clamp(0, 255)
        batch_data = batch_data.byte()  # Convert to uint8 for fid
        fake_data = fake_data.byte()
        fid.update(batch_data, real=True)
        fid.update(fake_data, real=False)
        batch_fid = fid.compute()
        fids.append(batch_fid.item())
        num_generated += fid_batch
        if num_generated > 100: break

    return fids

def hist_uv_kl(generator, dataloader, kl_batch=8, num_gen_layers=5, latent_dim=512, mixing_prob=0.9, device="cpu"):
    kls = []
    num_generated = 0
    for batch_data in dataloader:
        batch_data = batch_data.to(device)
        z = mixing_noise(kl_batch, num_gen_layers, latent_dim, mixing_prob).to(device) 
        target_hist = random_interpolate_hists(batch_data)
        fake_data, _ = generator(z, target_hist, test=True)
        relu_fake = torch.nn.functional.relu(fake_data, inplace=False)
        relu_fake = torch.clamp(relu_fake, 0, 1) # Fix relu inplace gradient
        fake_hist = histogram_feature_v2(relu_fake, device=device)
        target_hist /= torch.linalg.norm(target_hist)
        fake_hist /= torch.linalg.norm(fake_hist)
        kl = (target_hist*(target_hist.log()-fake_hist.log())).sum()/kl_batch  # Compute KL Div
        kls.append(kl.detach().numpy())
        num_generated += kl_batch
        if num_generated > 100: break
    
    return kls

def hist_uv_h(generator, dataloader, h_batch=8, num_gen_layers=5, latent_dim=512, mixing_prob=0.9, device="cpu"):
    hs = []
    num_generated = 0
    for batch_data in dataloader:
        batch_data = batch_data.to(device)
        z = mixing_noise(h_batch, num_gen_layers, latent_dim, mixing_prob).to(device) 
        target_hist = random_interpolate_hists(batch_data)
        fake_data, _ = generator(z, target_hist, test=True)
        h = hellinger_dist_loss(fake_data, target_hist, device=device)
        hs.append(h.detach().numpy())       
        num_generated += h_batch
        if num_generated > 100: break
    
    return hs

def interpret(generator, color_img_path, image_res, h_batch=5, num_gen_layers=5, latent_dim=512, mixing_prob=0.9, device="cpu"):
    transform = Resize((image_res, image_res))
    color_img = read_image(color_img_path).to(device).float()
    color_img = color_img/255.0
    color_img = transform(color_img)
    color_img = color_img.unsqueeze(dim=0)

    color_img = color_img.repeat((h_batch, 1, 1, 1))
    target_hist = histogram_feature_v2(color_img, device=device)
    z = mixing_noise(h_batch, num_gen_layers, latent_dim, mixing_prob).to(device)
    w = generator.get_w_from_z(z)
    fake_imgs = generator.gen_image_from_w(w, target_hist, test=True) 
    return fake_imgs, color_img
