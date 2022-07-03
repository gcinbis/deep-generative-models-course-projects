from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flow_modules.common_modules import Actnormlayer
from flow_modules.misc import cpd_sum, cpd_mean

class Conv2dZeros(nn.Conv2d):
	def __init__(self, in_channels, out_channels,
				 kernel_size=[3, 3], stride=[1, 1],
				 padding="same", logscale_factor=3):
		padding = Conv2d.get_padding(padding, kernel_size, stride)
		super(Conv2dZeros, self).__init__(in_channels, out_channels, kernel_size, stride, padding)
		# logscale_factor
		self.logscale_factor = logscale_factor
		self.register_parameter("logs", nn.Parameter(torch.zeros(out_channels, 1, 1)))
		# init
		self.weight.data.zero_()
		self.bias.data.zero_()

	def forward(self, input):
		output = super(Conv2dZeros, self).forward(input)
		return output * torch.exp(self.logs * self.logscale_factor)		

class Conv2d(nn.Conv2d):
	pad_dict = {
		"same": lambda kernel, stride: [((k - 1) * s + 1) // 2 for k, s in zip(kernel, stride)],
		"valid": lambda kernel, stride: [0 for _ in kernel]
	}

	@staticmethod
	def get_padding(padding, kernel_size, stride):
		# make paddding
		if isinstance(padding, str):
			if isinstance(kernel_size, int):
				kernel_size = [kernel_size, kernel_size]
			if isinstance(stride, int):
				stride = [stride, stride]
			padding = padding.lower()
			try:
				padding = Conv2d.pad_dict[padding](kernel_size, stride)
			except KeyError:
				raise ValueError("{} is not supported".format(padding))
		return padding

	def __init__(self, in_channels, out_channels,
				 kernel_size=[3, 3], stride=[1, 1],
				 padding="same", do_actnorm=True, weight_std=0.05):
		padding = Conv2d.get_padding(padding, kernel_size, stride)
		super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
						 padding, bias=(not do_actnorm))
		# init weight with std
		self.weight.data.normal_(mean=0.0, std=weight_std)
		if not do_actnorm:
			self.bias.data.zero_()
		else:
			self.actnorm = Actnormlayer(out_channels)
		self.do_actnorm = do_actnorm

	def forward(self, input):
		x = super(Conv2d, self).forward(input.type(torch.FloatTensor).cuda())
		if self.do_actnorm:
			x, _ = self.actnorm(x,0.0)
		return x

class NN_net(nn.Module):

	def __init__(self, in_channels, out_channels, hiddden_channels):
		super(NN_net, self).__init__()
		self.conv1 = Conv2d(in_channels, hiddden_channels)
		self.conv2 = Conv2d(hiddden_channels, hiddden_channels, kernel_size=[1, 1])
		self.conv3 = Conv2dZeros(hiddden_channels, out_channels)
	
	def forward(self,x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = self.conv3(x)
		return x


def split_feature(tensor, _type="split"):
	"""
	type = ["split", "cross"]
	"""
	C = tensor.size(1)
	if _type == "split":
		return tensor[:, :C // 2, ...], tensor[:, C // 2:, ...]
	elif _type == "cross":
		return tensor[:, 0::2, ...], tensor[:, 1::2, ...]


class AffineCoupling(nn.Module):
	def __init__(self, in_channels, out_channels, hiddden_channels):
		super(AffineCoupling, self).__init__()
		self.NN_net = NN_net(in_channels//2, out_channels, hiddden_channels)

	def split(self,x,_type="split"):
		return split_feature(x,_type)


	def forward_inference(self,x,logdet):
		z1, z2 = self.split(x)
		y2 = self.NN_net(z1)
		shift, scale = self.split(y2,"cross")
		scale = torch.sigmoid(scale + 2.0).to(x.device)
		z2	= z2*scale
		z2	    = shift.to(x.device) + z2
		logdet = cpd_sum(torch.log(scale), dim=[1, 2, 3]) + logdet
		z      = torch.cat((z1, z2), dim=1)
		return z, logdet

	def reverse_sampling(self,x,logdet):
		z1,z2  	= self.split(x)
		y2     	= self.NN_net(z1)
		shift, scale 	= self.split(y2,"cross")
		scale 		= torch.sigmoid(scale + 2.0)
		z2	    = z2-shift
		z2		= z2/scale
		logdet = logdet - cpd_sum(torch.log(scale), dim=[1, 2, 3])
		z = torch.cat((z1, z2), dim=1)
		return z, logdet

	def forward(self, input, logdet = 0., reverse=False):
		if not reverse:
			x, logdet = self.forward_inference(input, logdet)
		else:
			x, logdet = self.reverse_sampling(input, logdet)
		return x, logdet



class Split2d(nn.Module):
	def __init__(self, num_channels):
		super().__init__()
		self.conv = Conv2dZeros(num_channels // 2, num_channels)

	def split_feature(self, z):
		return z[:,:z.size(1)//2,:,:], z[:,z.size(1)//2:,:,:]

	def split2d_prior(self, z):
		h = self.conv(z)
		return h[:,0::2,:,:], h[:,1::2,:,:]

	def forward(self, input, logdet=0., reverse=False, eps_std=None):
		if not reverse:
			z1, z2 = self.split_feature(input)
			mean, logs = self.split2d_prior(z1)
			logdet = GaussianDiag.logp(mean, logs, z2) + logdet
			return z1, logdet
		else:
			z1 = input
			mean, logs = self.split2d_prior(z1)
			z2 = GaussianDiag.sample(mean, logs, eps_std)
			z = torch.cat((z1, z2), dim=1)
			return z, logdet		