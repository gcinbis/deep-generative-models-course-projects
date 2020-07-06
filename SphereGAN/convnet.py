import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model_utils import *

class GeneratorConvNet(nn.Module):
	'''
		ConvNet generator that works with hard-coded dimensions, where images are of size 48x48:
			
		Dimension flow (features, width, height):
			128 => 512,6,6 => 256,12,12 => 128,24,24 => 64,48,48 => 3,48,48 

	'''
	def __init__(self,):
		super(GeneratorConvNet, self).__init__()

		self.ln1 = nn.Linear(128, 6*6*512, bias=False)
		self.reshape = View((512, 6, 6))
		self.bn = nn.BatchNorm2d(512)
		self.relu = nn.ReLU(True)

		self.conv1 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)
		self.bn1 = nn.BatchNorm2d(256)
		self.relu1 = nn.ReLU(True)
		
		self.conv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
		self.bn2 = nn.BatchNorm2d(128)
		self.relu2 = nn.ReLU(True)

		self.conv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
		self.bn3 = nn.BatchNorm2d(64)
		self.relu3 = nn.ReLU(True)

		self.conv4 = nn.Conv2d(64, 3, 3, 1, 1, bias=False)
		self.tanh = nn.Tanh()

	def forward(self, input):
		output = self.ln1(input)
		output = self.reshape(output)
		output = self.bn(output)
		output = self.relu(output)

		output = self.conv1(output)
		output = self.bn1(output)
		output = self.relu1(output)

		output = self.conv2(output)
		output = self.bn2(output)
		output = self.relu2(output)

		output = self.conv3(output)
		output = self.bn3(output)
		output = self.relu3(output)

		output = self.conv4(output)
		output = self.tanh(output)

		return output

class DiscriminatorConvNet(nn.Module):
	'''
		A Convnet discriminator with hard-coded dimensions, plus the Geometric Block.
		
		Dimension flow (features, width, height):
			3,48,48 => 64,24,24 => 128,12,12 => 256,6,6 => 512,6,6 
	'''
	def __init__(self,):
		super(DiscriminatorConvNet, self).__init__()

		self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
		self.ln1_1 =  nn.LayerNorm([64,48,48])
		self.lrelu1_1 = nn.LeakyReLU(0.2, inplace=True)
		
		self.conv1_2 = nn.Conv2d(64, 64, 4, 2, 1, bias=False)
		self.ln1_2 =  nn.LayerNorm([64,24,24]) 
		self.lrelu1_2 = nn.LeakyReLU(0.2, inplace=True)


		self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1, bias=False)
		self.ln2_1 = nn.LayerNorm([128,24,24]) 
		self.lrelu2_1 = nn.LeakyReLU(0.2, inplace=True)
		
		self.conv2_2 = nn.Conv2d(128, 128, 4, 2, 1, bias=False)
		self.ln2_2 = nn.LayerNorm([128,12,12])
		self.lrelu2_2 = nn.LeakyReLU(0.2, inplace=True)


		self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
		self.ln3_1 = nn.LayerNorm([256,12,12]) 
		self.lrelu3_1 = nn.LeakyReLU(0.2, inplace=True)
		
		self.conv3_2 = nn.Conv2d(256, 256, 4, 2, 1, bias=False)
		self.ln3_2 = nn.LayerNorm([256,6,6]) 
		self.lrelu3_2 = nn.LeakyReLU(0.2, inplace=True)

		self.conv4 = nn.Conv2d(256, 512, 3, 1, 1, bias=False)
		self.gb = GeometricBlock(dim=512, pool=True)



	def forward(self, input):
		
		output = self.conv1_1(input)
		output = self.ln1_1(output)
		output = self.lrelu1_1(output)
	   
		output = self.conv1_2(output)
		output = self.ln1_2(output)
		output = self.lrelu1_2(output)

		output = self.conv2_1(output)
		output = self.ln2_1(output)
		output = self.lrelu2_1(output)
		
		output = self.conv2_2(output)
		output = self.ln2_2(output)
		output = self.lrelu2_2(output)

		output = self.conv3_1(output)
		output = self.ln3_1(output)
		output = self.lrelu3_1(output)
		
		output = self.conv3_2(output)
		output = self.ln3_2(output)
		output = self.lrelu3_2(output)

		output = self.conv4(output)
		output = self.gb(output)
		

		return output
