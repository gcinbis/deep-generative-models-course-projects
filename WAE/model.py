# -*- coding: utf-8 -*-
# --------------------------------------------------
#
# model.py
#
# Written by aliabbasi -*- ali.abbasi@metu.edu.tr
# Written by cetinsamet -*- cetin.samet@metu.edu.tr
# May, 2020
# --------------------------------------------------
import torch.nn.functional as F
import torch.nn as nn
import torch

import numpy as np


class Encoder(nn.Module):
	"""Encoder Network"""

	def __init__(self, h_dim, z_dim, n_channel, kernel_size):

		super(Encoder, self).__init__()

		# set params	
		self.h_dim = h_dim
		self.z_dim = z_dim
		self.n_channel = n_channel
		self.kernel_size = kernel_size

		# Module 1
		self.conv1 = nn.Conv2d(n_channel, self.h_dim, self.kernel_size, stride=2, padding=2) 
		self.bn1 = nn.BatchNorm2d(self.h_dim)
		# Module 2
		self.conv2 = nn.Conv2d(self.h_dim, 2 * self.h_dim, self.kernel_size, stride=2, padding=2)
		self.bn2 = nn.BatchNorm2d(2*self.h_dim)
		# Module 3
		self.conv3 = nn.Conv2d(2 * self.h_dim, 4 * self.h_dim, self.kernel_size, stride=2, padding=2)
		self.bn3 = nn.BatchNorm2d(4*self.h_dim)
		# Module 4
		self.conv4 = nn.Conv2d(4 * self.h_dim, 8 * self.h_dim, self.kernel_size, stride=2, padding=2)
		self.bn4 = nn.BatchNorm2d(8 * self.h_dim)
		# output layer
		self.out = nn.Linear(8 * self.h_dim * 4 * 4, self.z_dim) 

	def forward(self, x):
		# forward through Module 1
		x = self.conv1(x)
		x = self.bn1(x)
		x = F.relu(x)
		# forward through Module 2
		x = self.conv2(x)
		x = self.bn2(x)
		x = F.relu(x)
		# forward through Module 3
		x = self.conv3(x)
		x = self.bn3(x)
		x = F.relu(x)
		# forward through Module 4
		x = self.conv4(x)
		x = self.bn4(x)
		x = F.relu(x)
		# forward through output layer
		x = x.view(-1, 8 * self.h_dim * 4 * 4)
		x = self.out(x)

		return x

class Decoder(nn.Module):
	"""Decoder Network"""

	def __init__(self, h_dim, z_dim, n_channel, kernel_size):

		super(Decoder, self).__init__()

		self.h_dim = h_dim
		self.z_dim = z_dim
		self.n_channel = n_channel
		self.kernel_size = kernel_size

		# input layer
		self.inp = nn.Linear(self.z_dim, 8 * 8 * (8 * self.h_dim))
		# Module 1
		self.conv1 = nn.ConvTranspose2d(8 * self.h_dim, 4 * self.h_dim, self.kernel_size, stride=2, padding=2, output_padding=1)
		self.bn1 = nn.BatchNorm2d(4 * self.h_dim)
		# Module 2
		self.conv2 = nn.ConvTranspose2d(4 * self.h_dim, 2 * self.h_dim, self.kernel_size, stride=2, padding=2, output_padding=1)
		self.bn2 = nn.BatchNorm2d(2 * self.h_dim)
		# Module 3
		self.conv3 = nn.ConvTranspose2d(2 * self.h_dim, self.h_dim, self.kernel_size, stride=2, padding=2, output_padding=1)
		self.bn3 = nn.BatchNorm2d(self.h_dim)
		# output layer
		self.conv4 = nn.ConvTranspose2d(self.h_dim, self.n_channel, self.kernel_size, stride=1, padding=2)
				
	def forward(self, x):
		# forward through input layer
		x = self.inp(x)
		x = x.view(-1, 8 * self.h_dim, 8, 8)
		# forward through Module 1
		x = self.conv1(x)
		x = self.bn1(x)
		x = F.relu(x)
		# forward through Module 2
		x = self.conv2(x)
		x = self.bn2(x)
		x = F.relu(x)
		# forward through Module 3
		x = self.conv3(x)
		x = self.bn3(x)
		x = F.relu(x)
		# forward through output layer
		x = self.conv4(x)
		x = torch.tanh(x)

		return x
