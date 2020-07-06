# -*- coding: utf-8 -*-
# --------------------------------------------------
#
# generate.py
#
# Written by aliabbasi -*- ali.abbasi@metu.edu.tr
# Written by cetinsamet -*- cetin.samet@metu.edu.tr
# May, 2020
# --------------------------------------------------
from tool import *

import numpy as np
import torch


# -------------------------------------------------------------- #
# RECONSTRUCTION
# -------------------------------------------------------------- #
def reconstruction(x, encoder, decoder, device=torch.device('cpu')):
	
	# image
	x = x.to(device)

	# encode -- P(z|x)
	z = encoder(x)
	# decode -- P(x|z)
	x_recon = decoder(z).detach()
    
	# denormalize images
	x = denormalize(x).cpu()
	x_recon = denormalize(x_recon).cpu()
    
	return x, x_recon

# -------------------------------------------------------------- #
# INTERPOLATION
# -------------------------------------------------------------- #
def interpolation(x1, x2, encoder, decoder, n_step=8, device=torch.device('cpu')):
	
	# images
	x1 = x1.to(device) # 1st image
	x2 = x2.to(device) # 2nd image

	# encode -- P(z|x)
	z1 = encoder(x1) # latent code of the 1st image
	z2 = encoder(x2) # latent code of the 2nd image 
    
	x_inter_codes = []
	for ratio in torch.linspace(0, 1, n_step):
		# calculate interpolated latent codes (z)
		z_inter_code = (1 - ratio) * z1 + ratio * z2
		# decode -- P(x|z)
		x_inter_code = decoder(z_inter_code).detach()
		x_inter_codes.append(x_inter_code)
	x_inter_codes = torch.cat([x1] + x_inter_codes + [x2])

	# denormalize images
	x_inter_codes = denormalize(x_inter_codes).cpu()
	
	return x_inter_codes

# -------------------------------------------------------------- #
# RANDOM-SAMPLING
# -------------------------------------------------------------- #
def random_sampling(decoder, n_sample, z_dim=64, device=torch.device('cpu')):
	
	# generate a random noise from prior normal distribution
	z = torch.randn(n_sample, z_dim).to(device)
	
	# decode -- P(x|z)
	x_sampled = decoder(z).detach()
	
	# denormalize image
	x_sampled = denormalize(x_sampled).cpu()
	
	return x_sampled
