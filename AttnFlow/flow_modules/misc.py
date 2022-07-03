from __future__ import print_function
import random
import numpy as np
import torch
import torch.nn as nn

import torchvision.transforms.functional as TF

def cpd_sum(tensor, dim=None, keepdim=False):
	if dim is None:
		# sum up all dim
		return torch.sum(tensor)
	else:
		if isinstance(dim, int):
			dim = [dim]
		dim = sorted(dim)
		for d in dim:
			tensor = tensor.sum(dim=d, keepdim=True)
		if not keepdim:
			for i, d in enumerate(dim):
				tensor.squeeze_(d-i)
		return tensor

def cpd_mean(tensor, dim=None, keepdims=False):
	if dim is None:
		return tensor.mean(tensor)
	else:
		if isinstance(dim, int):
			dim = [dim]
		dim = sorted(dim)
		for d in dim:
			tensor = tensor.mean(dim=d, keepdim=True)
		if not keepdims:
			for i, d in enumerate(dim):
				tensor.squeeze_(d-i)
		return tensor	

class ShiftTransform:

	def __init__(self, pixels):
		self.pixels = pixels
		self.pixel_shifts = [i for i in range(0,2*self.pixels)]

	def __call__(self, x):
		width, height = x.size
		shift = int(random.choice(self.pixel_shifts))
		x = TF.pad(x, self.pixels, padding_mode='edge')
		if random.random() < 0.5:
			return x.crop(box=(shift,self.pixels,shift+width,self.pixels+height))
		else:
			return x.crop(box=(self.pixels,shift,self.pixels+width,shift+height))

class MnistGlowTransform:

	def __init__(self, pixels):
		self.pixels = pixels
		#self.pixel_shifts = [i for i in range(0,2*self.pixels)]

	def __call__(self, x):
		x = TF.pad(x, self.pixels)
		x = np.array(x)
		x = np.concatenate([x[:,None,:,:],x[:,None,:,:],x[:,None,:,:]], axis=1)
		return Image.fromarray(x)