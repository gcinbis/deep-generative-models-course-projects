import json
from collections import OrderedDict
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.optim as optim

import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from convnet import DiscriminatorConvNet, GeneratorConvNet
from resnet import DiscriminatorResNet, GeneratorResNet

from inception_score import *
from frechet_inception_distance import *


def get_stl10_dataloader_save(dataroot, batch_size, workers, image_size=48):
	'''
		Returns a loader for unlabeled images that will be used in IS and FID calculation.
	'''
	unl_set = dset.STL10(root=dataroot, download=True, split='unlabeled',
					transform=transforms.Compose([
					   transforms.Resize(image_size),
					   transforms.ToTensor(),
					   ]))

	loader = torch.utils.data.DataLoader(
						unl_set, 
						batch_size=batch_size, shuffle=True,
						num_workers=workers, pin_memory=False)

	return loader

def save_stl10_images(dataroot, folder, num_images=1024):
	'''
		Using unlabeled dataloader, this one save images to given folder to be used in IS and FID calculation.

	'''
	if not os.path.exists(folder):
		os.makedirs(folder)
	

	batch_size = 64
	dataloader = get_stl10_dataloader_save(dataroot, batch_size=batch_size, workers=4, image_size=48)

	idx = 0
	for i, data in enumerate(dataloader, 0):
		
		if i*batch_size >= num_images:
			break

		image_batch = data[0]
		for image in image_batch:
			image_path = os.path.join(folder, "sample_image_{}.png".format(idx))
			save_image(image, image_path)
			idx += 1



def get_stl10_dataloader(dataroot, batch_size, workers, image_size=48):
	'''
		Returns a loader for training images that will be used in training.

	'''
	train_set = dset.STL10(root=dataroot, download=True,
					transform=transforms.Compose([
					   transforms.Resize(image_size),
					   transforms.ToTensor(),
					   transforms.Normalize((0.5, 0.5, 0.5), (0.511, 0.511, 0.511)),
					   ]))

	train_loader = torch.utils.data.DataLoader(
						train_set,
						batch_size=batch_size, shuffle=True,
						num_workers=workers, pin_memory=False, drop_last=True)

	return train_loader


def save_model(model, save_path):
	torch.save(model, save_path)

def load_model(model_path):
	model = torch.load(model_path)
	return model

def save_state_dict(gen_model, disc_model, optim_gen, optim_disc):

	PATH = './checkpoints/temp.pt'

	torch.save({
				'gen_state_dict': gen_model.state_dict(),
				'gen_optimizer_state_dict': optim_gen.state_dict(),
				'dis_state_dict': disc_model.state_dict(),
				'dis_optimizer_state_dict': optim_disc.state_dict(),
				}, PATH)

def load_state_dict(mode="resnet"):

	state_dict = torch.load("./checkpoints/temp.pt")

	gen = None
	disc = None
	
	if mode.lower() == "resnet":
		gen = GeneratorConvNet()
		disc = DiscriminatorConvNet()
	elif mode.lower() == "convnet":
		gen = GeneratorResNet()
		disc = DiscriminatorResNet()
	else:
		raise RuntimeError("invalid mode")

	gen.load_state_dict(state_dict["gen_state_dict"])
	disc.load_state_dict(state_dict["dis_state_dict"])

	optimizerGen = optim.Adam(gen.parameters(), lr=lr, betas=(beta1, beta2))
	optimizerDisc = optim.Adam(dis.parameters(), lr=lr, betas=(beta1, beta2))

	optimizerGen.load_state_dict(state_dict["gen_optimizer_state_dict"])
	optimizerDisc.load_state_dict(state_dict["dis_optimizer_state_dict"])

	return gen, disc, optimizerGen, optimizerDisc


def plot_loss(G_losses, D_losses, save_path):
	plt.clf()
	plt.figure(figsize=(10,5))
	plt.title("Generator and Discriminator Loss During Training")
	plt.plot(G_losses,label="G")
	plt.plot(D_losses,label="D")
	plt.xlabel("iterations")
	plt.ylabel("Loss")
	plt.legend()
	plt.savefig(save_path)
	plt.clf()


def generate_images(model, num_samples, device, save_path):
	'''
		generates images with the given model.
	'''
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	batch_size = 64
	fixed_noise = torch.randn(num_samples, 128, device="cpu")
	model = model.to(device)

	img_list = []
	mean = torch.from_numpy(np.array([-1.0, -1.0, -1.0], dtype=np.float32)[:, np.newaxis, np.newaxis]).to(device)
	std = torch.from_numpy(np.array([2.0, 2.0, 2.0], dtype=np.float32)[:, np.newaxis, np.newaxis]).to(device)

	idx = 0
	for i in range(0, num_samples, batch_size):
		start = i
		end = i + batch_size

		part = fixed_noise[start:end].to(device)

		output = model(part)

		output -= mean
		output /= std
   
		for image in output:
			image_path = os.path.join(save_path, "sample_image_{}.png".format(idx))
			save_image(image, image_path)
			if idx < 64:
				img_list.append(image.detach().cpu().numpy())
			idx += 1


	img_list = np.array(img_list)[:64]
	plt.figure(figsize=(15,15))
	plt.title("Generated Images")
	plt.imshow(np.transpose(vutils.make_grid(torch.from_numpy(img_list).to(device), padding=5, normalize=True).cpu(),(1,2,0)))

	arch = save_path.split("_")[-1]
	plt.savefig("./figures/generated_images_{}.svg".format(arch))

def eval_model(model, dataroot, num_samples, device, save_path):
	'''
		This one unifies process of qualitative and quantitative evaluation
	'''
	batch_size = 8

	print("Generating images...")
	generate_images(model, num_samples, device, save_path)

	real_save_path = "./stl_test_samples"
	print("Saving real images...")
	save_stl10_images(dataroot, real_save_path, num_samples)

	print("Calculating IS...")
	# Calculate the IS score using only the generated images.
	is_mean, is_std = calculate_is_score_given_path(save_path, batch_size, device)

	print("Calculating FID...")
	# Calculate the FID scores using real and generated images.
	paths = [save_path, real_save_path]
	fid = calculate_fid_given_paths(paths, batch_size, device)

	print("IS:", is_mean, is_std)
	print("FID:", fid)

	return is_mean, is_std, fid
