import torch
from torchvision import transforms
import scipy.linalg
import numpy as np
import torchvision

# FID calculations

def fid(real_feats, generated_feats):

	mean_real = real_feats.mean(dim=0)
	mean_generated = generated_feats.mean(dim=0)

	cov_real = torch.from_numpy(np.cov(real_feats.detach().cpu().numpy(), rowvar=False)).to(real_feats.device)
	cov_generated = torch.from_numpy(np.cov(generated_feats.detach().cpu().numpy(), rowvar=False)).to(real_feats.device)

	squared_diff = torch.pow((mean_real - mean_generated),2).sum()
	dotprod = torch.mm(cov_real, cov_generated)
	squared_dotprod = torch.from_numpy(scipy.linalg.sqrtm(dotprod.detach().cpu().numpy().astype(np.float32))).to(real_feats.device)
	if torch.is_complex(squared_dotprod):
		squared_dotprod = squared_dotprod.real

	calc_trace = torch.trace(cov_real + cov_generated - 2.0*squared_dotprod)

	return squared_diff + calc_trace

def calculate_fid(gen_path, batch_size = 32, cuda = True):
	#Path determines whether it is a SNGAN Generator or a SNGAN-aw Generator
	import models

	torch.manual_seed(0)

	image_transforms = transforms.Compose(
	                    [transforms.ToTensor(),
	                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
	                                        download=True, transform=image_transforms)

	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
	                                          shuffle=True, num_workers=8)

	preprocess = transforms.Compose([
			    transforms.Resize(299),
			    transforms.CenterCrop(299),
			    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
				])

	inception = torch.hub.load('pytorch/vision:v0.8.0', 'inception_v3', pretrained=True)
	inception = inception.cuda() if cuda else inception
	inception.fc = torch.nn.Identity()
	inception.eval()

	generator = models.SNGAN_Generator()
	generator.load_state_dict(torch.load(gen_path)) # Load model
	generator = generator.cuda() if cuda else generator

	generator.eval()
	generated_feats = torch.zeros(len(trainset), 2048).to(next(generator.parameters()).device)
	real_feats = torch.zeros(len(trainset), 2048).to(next(generator.parameters()).device)
	for iter_idx, data in enumerate(trainloader): 
		data, labels = data
		data = data.cuda() if cuda else data
		noise = torch.randn(data.size(0), 128).to(data.device)
		generated = generator(noise)

		generated = (generated + 1) / 2
		data = (data + 1) / 2

		generated = preprocess(generated) 
		data = preprocess(data)


		with torch.no_grad():
			generated_feats[iter_idx*data.size(0):(iter_idx+1)*data.size(0)] = inception(generated)
			real_feats[iter_idx*data.size(0):(iter_idx+1)*data.size(0)] = inception(data)

	fid_score = fid(real_feats, generated_feats)
	print("Epoch %d FID %.2f"%(epoch, fid_score))
