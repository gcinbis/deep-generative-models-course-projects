# -*- coding: utf-8 -*-
# --------------------------------------------------
#
# kernel.py
#
# Written by aliabbasi -*- ali.abbasi@metu.edu.tr
# Written by cetinsamet -*- cetin.samet@metu.edu.tr
# May, 2020
# --------------------------------------------------
import torch.nn.functional as F
import torch


def IMQ(x, y, z_dim, sigma, device):
	"""IMQ kernel
	
	Arguments:
	x -- encoded latent code, P(z|x)
	y -- randomly generated latent code from prior, P(z)
	device -- 'cpu' or 'cuda'
	"""
	
	# encoded -- P(z|x)
	x_norm = torch.sum(x ** 2, dim=1, keepdim=True)
	x_dotprod = torch.mm(x, torch.t(x))
	x_dist = x_norm + torch.t(x_norm) - 2. * x_dotprod
	
	# randomly selected from prior -- P(z)
	y_norm = torch.sum(y ** 2, dim=1, keepdim=True)
	y_dotprod = torch.mm(y, torch.t(y))
	y_dist = y_norm + torch.t(y_norm) - 2. * y_dotprod
	
	# joint
	dotprod = torch.mm(x, torch.t(y))
	dist = x_norm + torch.t(y_norm) - 2. * dotprod
	
	batch_size = x.shape[0]
	c = 2. * z_dim * sigma
	
	res1 = (c / (c + y_dist)) + (c / (c + x_dist))
	res1 = res1 * (1. - torch.eye(batch_size)).to(device)
	res1 = torch.sum(res1) / ((batch_size ** 2) - batch_size)
		
	res2 = c / (c + dist)
	res2 = torch.sum(res2) * 2. / (batch_size ** 2)
		
	stat = res1 - res2
	
	return stat
