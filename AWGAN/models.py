import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

# This implementation directly uses SNGAN ResNet Generator and Discriminator

class ResNetBlock(nn.Module):
	def __init__(self, in_channel, out_channel, block_type = "generator", blockno=0):
		super(ResNetBlock, self).__init__()
		self.bn1 = nn.BatchNorm2d(in_channel)
		self.act1 = nn.ReLU()
		self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1) if block_type == "generator" else spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1))
		self.bn2 = nn.BatchNorm2d(out_channel)
		self.act2 = nn.ReLU()
		self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1) if block_type == "generator" else spectral_norm(nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1))
		self.conv_residual = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0) if block_type == "generator" else spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0))
		self.block_type = block_type
		self.blockno = blockno
	
	def forward(self, x):
		b_size, channel, h, w = x.size()
		if self.block_type == "generator":
			x = self.bn1(x)
		out = self.act1(x)
		out = self.conv1(out)
		if self.block_type == "generator":
			out = F.interpolate(out, size=(h*2,w*2), mode="bilinear")
			out = self.bn2(out)
		out = self.act2(out)
		out = self.conv2(out)
		if (self.block_type == "discriminator") and (self.blockno == 0 or self.blockno == 1):
			out = F.avg_pool2d(out, 2)

		residual = self.conv_residual(x)
		if self.block_type == "generator":
			residual = F.interpolate(residual, size=(h*2,w*2), mode="bilinear")

		elif (self.block_type == "discriminator") and (self.blockno == 0 or self.blockno == 1):	
			residual = F.avg_pool2d(residual, 2)

		return residual + out

class SNGAN_Generator(nn.Module):
	def __init__(self):
		super(SNGAN_Generator, self).__init__()
		self.linear = nn.Linear(128, 4*4*256) # z = [batch, 128] -> [batch, 256, 4, 4]
		resblocks = []
		resblocks += [ResNetBlock(256, 256, block_type="generator")]
		resblocks += [ResNetBlock(256, 256, block_type="generator")]
		resblocks += [ResNetBlock(256, 256, block_type="generator")]
		self.resblocks = nn.Sequential(*resblocks)
		self.bn = nn.BatchNorm2d(256)
		self.act = nn.ReLU()
		self.conv = nn.Conv2d(256, 3, kernel_size=3, padding=1)
		self.tanh = nn.Tanh()

	def forward(self, z):
		b_size, z_dim = z.size()
		out = self.linear(z)
		out = out.view(b_size, 256, 4, 4)
		out = self.resblocks(out)
		out = self.conv(self.act(self.bn(out)))
		return self.tanh(out)

class SNGAN_Discriminator(nn.Module):
	def __init__(self):
		super(SNGAN_Discriminator, self).__init__()
		resblocks = []
		resblocks += [ResNetBlock(3, 128, block_type="discriminator", blockno=0)]
		resblocks += [ResNetBlock(128, 128, block_type="discriminator", blockno=1)]
		resblocks += [ResNetBlock(128, 128, block_type="discriminator", blockno=2)]
		resblocks += [ResNetBlock(128, 128, block_type="discriminator", blockno=3)]
		self.resblocks = nn.Sequential(*resblocks)
		self.act = nn.ReLU()
		self.linear = spectral_norm(nn.Linear(128, 1))

	def forward(self, x):	
		out = self.resblocks(x)
		out = self.act(out)
		out = F.avg_pool2d(out, out.size(2))
		out = out.view(out.size(0), -1)
		return self.linear(out)
