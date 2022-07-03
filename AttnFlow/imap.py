import torch
from torch import nn
import math
import numpy as np
class IMAP(nn.Module):
	def __init__(self, input_channels):
		super(IMAP, self).__init__()
		self.input_channels = input_channels
		self.weight = torch.empty([self.input_channels, self.input_channels,1])
		nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
		self.weight = torch.nn.Parameter(self.weight).cuda()
		self.bias = torch.empty([self.input_channels])
		fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
		bound = 1 / math.sqrt(fan_in)
		nn.init.uniform_(self.bias, -bound, bound)
		#self.conv1d = nn.Conv1d(self.input_channels, self.input_channels, kernel_size=1, bias=True)
		self.bias = torch.nn.Parameter(self.bias)
		#self.conv1d.bias = self.bias
		self.weight = torch.nn.Parameter(self.weight)
		#self.conv1d.weight = self.weight
		self.register_parameter("s", nn.Parameter(torch.randn([1, self.input_channels, 1])))
		self.register_parameter("offset", nn.Parameter(torch.ones([1])*8))
		self.pool1 = torch.nn.AvgPool1d(self.input_channels)

	def forward(self, input: torch.Tensor, logdet=0, reverse=False, permute=False):
		if not reverse:
			self.num_channels = input.shape[-1]**2
			#checkerboard
			B, C, H, W = input.shape

			self.mask = torch.tensor(np.ones((B,C, H*W), dtype=np.float64)).to(input.device)

			ones = torch.tensor(np.ones((1, 1, H*W), dtype=np.float64))
			ones.flatten()[::2] = 0
			zeros = torch.tensor(np.zeros((1, 1, H*W), dtype=np.float64))
			zeros.flatten()[::2] = 1
			grid = torch.cat((ones, zeros))
			cat_ones = torch.cat((ones, zeros))

			for i in range(0, C//2 - 1):
				grid = torch.cat((grid, cat_ones))
			grid = grid.view(1,C,H*W)

			checkerboard = torch.cat((grid, grid))
			checkerboard_1 = torch.cat((grid, grid))

			for i in range(1, B//2):
				checkerboard = torch.cat((checkerboard, checkerboard_1))

			if B == 1:
				checkerboard = grid


			self.mask = checkerboard.to(input.device)

			sig = torch.nn.Sigmoid()
			input_masked = input.view(B, C, H*W) * self.mask
			z = torch.nn.functional.conv1d(input_masked.type(torch.FloatTensor).to(input.device), self.weight.to(input.device), self.bias.to(input.device))
				#self.conv1d(input_masked.type(torch.FloatTensor).cuda())
			z_new = z.transpose(1, 2)
			pool_out = self.pool1(z_new)
			attn_out = (sig(pool_out.squeeze(-1) + self.offset.to(input.device)) + 1e-5).unsqueeze(1)
			attn_mask = (1 - self.mask) * attn_out + self.mask * (sig(self.s.to(input.device)) + 1e-5)
			out_new = input * attn_mask.view(B, C, H*W).view(B, C, H, W)
			logdet = logdet + torch.sum((self.input_channels//2) * (torch.log(sig(pool_out.squeeze(-1)+ self.offset.to(input.device))+1e-5)), dim=-1)
			logdet = logdet + torch.sum(torch.log(sig(self.s.to(input.device))+1e-5) * self.mask)
			return out_new, logdet
		else:
			out_new = input
			self.num_channels = input.shape[-1]**2
			B, C, H, W = out_new.shape
			self.mask = torch.tensor(np.ones((B, C, H * W), dtype=np.float64)).to(input.device)

			ones = torch.tensor(np.ones((1, 1, H * W), dtype=np.float64))
			ones.flatten()[::2] = 0
			zeros = torch.tensor(np.zeros((1, 1, H * W), dtype=np.float64))
			zeros.flatten()[::2] = 1
			grid = torch.cat((ones, zeros))
			cat_ones = torch.cat((ones, zeros))

			for i in range(0, C // 2 - 1):
				grid = torch.cat((grid, cat_ones))
			grid = grid.view(1, C, H * W)

			checkerboard = torch.cat((grid, grid))
			checkerboard_1 = torch.cat((grid, grid))

			for i in range(1, B // 2):
				checkerboard = torch.cat((checkerboard, checkerboard_1))

			self.mask = checkerboard.to(input.device)
			sig = torch.nn.Sigmoid()
			s_sig = sig(self.s) + 1e-5
			s_sig_in = torch.ones_like(s_sig) / s_sig
			inp_masked = out_new.view(B, C, H*W) * self.mask * s_sig_in
			out_conv = torch.nn.functional.conv1d(inp_masked.type(torch.FloatTensor).to(input.device), self.weight, self.bias)
			pool_out = self.pool1(out_conv.transpose(1, 2))
			attn_out = (sig(pool_out.squeeze(2) + self.offset) + 1e-5).unsqueeze(1)
			attn_out = torch.ones_like(attn_out) / attn_out
			attn_mask = (1 - self.mask) * attn_out + self.mask * s_sig_in
			input_rev = out_new * (attn_mask.view(B, C, H*W).view(B, C, H, W)).type(torch.FloatTensor).to(input.device)
			logdet = logdet - torch.sum((self.input_channels//2) * (torch.log(sig(pool_out.squeeze(-1) + self.offset)+1e-5)), dim=-1)
			logdet = logdet - torch.sum(torch.log(sig(self.s) + 1e-5) * self.mask)
			return input_rev, logdet