# -*- coding: utf-8 -*-
# --------------------------------------------------
#
# tool.py
#
# Written by aliabbasi -*- ali.abbasi@metu.edu.tr
# Written by cetinsamet -*- cetin.samet@metu.edu.tr
# May, 2020
# --------------------------------------------------
from collections import OrderedDict
import numpy as np
import random
import torch


def set_seed(seed):
	""" Seed all random number generators """
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)  # for multiGPUs.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	
def set_device():
	"""set GPU/CPU device"""
	if torch.cuda.is_available(): 
		return torch.device('cuda') # CUDA is available!
	else: 
		return torch.device('cpu') # CUDA is NOT available

def denormalize(x):
	"""denormalize image

	CelebA images are mapped into [-1, 1] interval before training
	This function applies denormalization

	Arguments:
	x -- image
	"""
	x = x * .5 + .5
	return x

def from_parallel(path, device=torch.device('cpu')):
	
	# original saved file with DataParallel
	state_dict = torch.load(path, map_location=device)
	
	# create new OrderedDict that does not contain `module.`
	new_state_dict = OrderedDict()
	
	for k, v in state_dict.items():
		name = k[7:] # remove `module.`
		new_state_dict[name] = v
	
	return new_state_dict
	
