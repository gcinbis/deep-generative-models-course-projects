import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import models
import torch.nn.functional as F
from eval import fid
import numpy as np

# Training routine for SNGAN

def train_sngan(batch_size=32, lr=0.1, cuda=True, num_epoch=100, num_disc_updates=1):

	image_transforms = transforms.Compose(
	                    [transforms.ToTensor(),
	                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
	                                        download=True, transform=image_transforms)

	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
	                                          shuffle=True, num_workers=8)

	torch.manual_seed(0)

	generator = models.SNGAN_Generator().cuda() if cuda else models.SNGAN_Generator()
	discriminator = models.SNGAN_Discriminator().cuda() if cuda else models.SNGAN_Discriminator()

	inception = torch.hub.load('pytorch/vision:v0.8.0', 'inception_v3', pretrained=True)
	inception = inception.cuda() if cuda else inception
	inception.fc = torch.nn.Identity()
	inception.eval()

	optim_generator = optim.Adam(generator.parameters(), lr, betas=(0.5, 0.999))
	optim_discriminator = optim.Adam(discriminator.parameters(), lr, betas=(0.5, 0.999))

	preprocess = transforms.Compose([
			    transforms.Resize(299),
			    transforms.CenterCrop(299),
			    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
				])
	best_fid = np.inf

	for epoch in range(num_epoch):
		generator.train()
		discriminator.train()
		for iter_idx, data in enumerate(trainloader):
			data, labels = data
			data = data.cuda() if cuda else data
			#update discriminator
			optim_discriminator.zero_grad()
			noise = torch.randn(batch_size, 128).cuda() if cuda else torch.randn(batch_size, 128)
			generated = generator(noise)

			disc_real = F.relu(1 - discriminator(data)).mean()
			disc_fake = F.relu(1 + discriminator(generated)).mean()

			disc_loss = disc_real + disc_fake
			disc_loss.backward()
			optim_discriminator.step()

			if iter_idx % num_disc_updates == 0:
			# Update generator
				optim_generator.zero_grad()
				noise = torch.randn(batch_size, 128).cuda() if cuda else torch.randn(batch_size, 128)
				generated = generator(noise)
				gen_loss = -discriminator(generated).mean()
				gen_loss.backward()
				optim_generator.step()

			if iter_idx % 50 == 0:
				print("Epoch %d Iter %d Loss disc %.2f Loss gen %.2f"%(epoch, iter_idx, disc_loss.item(), gen_loss.item()))

		if epoch % 5 == 0:
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
		
		if fid_score < best_fid:
			best_fid = fid_score
			torch.save(generator.state_dict, "best_sngan_gen.ckpt")
			torch.save(discriminator.state_dict, "best_sngan_disc.ckpt")

	return generator, discriminator

