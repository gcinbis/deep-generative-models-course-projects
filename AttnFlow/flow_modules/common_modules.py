from __future__ import print_function
import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F

from flow_modules.misc import cpd_sum, cpd_mean

		

def squeeze2d(input, factor=2):
	#assert factor >= 1 and isinstance(factor, int)
	if factor == 1:
		return input
	size = input.size()
	B = size[0]
	C = size[1]
	H = size[2]
	W = size[3]
	assert H % factor == 0 and W % factor == 0, "{}".format((H, W))
	x = input.view(B, C, H // factor, factor, W // factor, factor)
	x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
	x = x.view(B, C * factor * factor, H // factor, W // factor)
	return x


def unsqueeze2d(input, factor=2):
	assert factor >= 1 and isinstance(factor, int)
	factor2 = factor ** 2
	if factor == 1:
		return input
	size = input.size()
	B = size[0]
	C = size[1]
	H = size[2]
	W = size[3]
	assert C % (factor2) == 0, "{}".format(C)
	x = input.view(B, C // factor2, factor, factor, H, W)
	x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
	x = x.view(B, C // (factor2), H * factor, W * factor)
	return x

class SqueezeLayer(nn.Module):
	def __init__(self, factor):
		super(SqueezeLayer, self).__init__()
		self.factor = factor

	def forward(self, input, logdet=0., reverse=False):
		if not reverse:
			output = squeeze2d(input, self.factor)
			return output, logdet
		else:
			output = unsqueeze2d(input, self.factor)
			return output, logdet					

class InvertibleConv1x1(nn.Module):
	def __init__(self, num_channels, LU_decomposed=True):
		super().__init__()
		w_shape = [num_channels, num_channels]
		w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
		if not LU_decomposed:
			# Sample a random orthogonal matrix:
			self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
		else:
			np_p, np_l, np_u = scipy.linalg.lu(w_init)
			np_s = np.diag(np_u)
			np_sign_s = np.sign(np_s)
			np_log_s = np.log(np.abs(np_s))
			np_u = np.triu(np_u, k=1)
			l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
			eye = np.eye(*w_shape, dtype=np.float32)

			self.register_buffer('p', torch.Tensor(np_p.astype(np.float32)))
			self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(np.float32)))
			self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)))
			self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)))
			self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)))
			self.l_mask = torch.Tensor(l_mask)
			self.eye = torch.Tensor(eye)
		self.w_shape = w_shape
		self.LU = LU_decomposed

	def get_weight(self, input, reverse):
		w_shape = self.w_shape
		pixels = list(input.size())[-1]
		if not self.LU:

			#thops.pixels(input)
			dlogdet = (torch.slogdet(self.weight)[1]) * pixels*pixels
			if not reverse:
				weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
			else:
				weight = torch.inverse(self.weight.double()).float()\
							  .view(w_shape[0], w_shape[1], 1, 1)
			return weight, dlogdet
		else:
			self.p = self.p.to(input.device)
			self.sign_s = self.sign_s.to(input.device)
			self.l_mask = self.l_mask.to(input.device)
			self.eye = self.eye.to(input.device)
			l = self.l.to(input.device) * self.l_mask + self.eye
			u = self.u.to(input.device) * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s.to(input.device)))
			dlogdet = cpd_sum(self.log_s) * pixels*pixels
			if not reverse:
				w = torch.matmul(self.p, torch.matmul(l, u))
			else:
				l = torch.inverse(l.cpu().double()).float()
				u = torch.inverse(u.cpu().double()).float()
				w = torch.matmul(u, torch.matmul(l, self.p.cpu().inverse())).cuda()
			return w.view(w_shape[0], w_shape[1], 1, 1).to(input.device), dlogdet.to(input.device)

	def forward(self, input, logdet=None, reverse=False):
		"""
		log-det = log|abs(|W|)| * pixels
		"""
		weight, dlogdet = self.get_weight(input, reverse)
		if not reverse:
			z = F.conv2d(input.type(torch.FloatTensor).to(input.device), weight)
			if logdet is not None:
				logdet = logdet + dlogdet.to(input.device)
			return z, logdet
		else:
			z = F.conv2d(input.type(torch.FloatTensor).cuda(), weight)
			if logdet is not None:
				logdet = logdet - dlogdet
			return z, logdet


class Actnormlayer(nn.Module):
	def __init__(self, num_features, scale=1.):
		super(Actnormlayer, self).__init__()
		self.register_buffer('is_initialized', torch.zeros(1))
		self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
		self.logs = nn.Parameter(torch.zeros(1, num_features, 1, 1))

		self.num_features = num_features
		self.scale = float(scale)
		self.eps = 1e-6

	def initialize_parameters(self, x):
		if not self.training:
			return

		with torch.no_grad():
			bias = -cpd_mean(x.clone(), dim=[0, 2, 3], keepdims=True)
			v = cpd_mean((x.clone() + bias) ** 2, dim=[0, 2, 3], keepdims=True)
			logs = (self.scale / (v.sqrt() + self.eps)).log()
			self.bias.data = bias.data.detach().clone()
			self.logs.data = logs.data.detach().clone()
			self.is_initialized += 1.

	def _center(self, x, reverse=False):
		if reverse:
			return x - self.bias.to(x.device)
		else:
			return x + self.bias.to(x.device)

	def _scale(self, x, sldj, reverse=False):
		logs = self.logs.to(x.device)
		if reverse:
			x = x * logs.mul(-1).exp()
		else:
			x = x * logs.exp()

		if sldj is not None:
			ldj = logs.sum() * x.size(2) * x.size(3)
			if reverse:
				sldj = sldj - ldj
			else:
				sldj = sldj + ldj

		return x, sldj

	def forward(self, x, ldj=None, reverse=False):
		if not self.is_initialized:
			self.initialize_parameters(x)

		if reverse:
			x, ldj = self._scale(x, ldj, reverse)
			x = self._center(x, reverse)
		else:
			x = self._center(x, reverse)
			x, ldj = self._scale(x, ldj, reverse)

		return x, ldj


class Split2dMsC(nn.Module):
	def __init__(self, num_channels, level=0):
		super().__init__()
		self.level = level

	def split_feature(self, z):
		return z[:,:z.size(1)//2,:,:], z[:,z.size(1)//2:,:,:]

	def split2d_prior(self, z):
		h = self.conv(z)
		return h[:,0::2,:,:], h[:,1::2,:,:]

	def forward(self, input, logdet=0., reverse=False, eps_std=None):
		if not reverse:
			z1, z2 = self.split_feature(input)
			return ( z1, z2), logdet
		else:
			z1, z2 = input
			z = torch.cat((z1, z2), dim=1)
			return z, logdet					

class TupleFlip(nn.Module):
	def __init__(self, ):
		super().__init__()

	def forward(self, z, logdet=0., reverse=False):
		if not reverse:
			z1, z2 = z.chunk(2, dim=1)
			return torch.cat([z2,z1], dim=1), logdet
		else:
			z2, z1 = z.chunk(2, dim=1)
			return torch.cat([z1,z2], dim=1), logdet


class GaussianDiag:
	Log2PI = float(np.log(2 * np.pi))

	@staticmethod
	def likelihood(mean, logs, x):
		return -0.5 * (logs * 2. + ((x - mean) ** 2) / torch.exp(logs * 2.) + GaussianDiag.Log2PI)

	@staticmethod
	def logp(mean, logs, x):
		likelihood = GaussianDiag.likelihood(mean, logs, x)
		return cpd_sum(likelihood, dim=[1,2,3])

	@staticmethod
	def sample(mean, logs, eps_std=None):
		eps_std = eps_std or 1
		eps = torch.normal(mean=torch.zeros_like(mean),
						   std=torch.ones_like(logs) * eps_std)
		return mean + torch.exp(logs) * eps	

		

