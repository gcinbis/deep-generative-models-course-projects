import torch
from model.generator import Generator
from model.discriminator import Discriminator
from loss.WGANGP import PG_Gradient_Penalty
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor
from os import getcwd
from numpy import array, log2
from time import time, sleep
import matplotlib.pyplot as plt

level_index = 0
config = {'channels':[128, 128, 128, 128, 128, 128], # must be len(config['sr]) + 1 
          'latent_size':128, 
          'sr':[4, 8, 16 , 32, 64], # spatial resolution
          'start_sr':64,
          'level_batch_size':[32, 32, 32 , 32, 16], 
          'epochs_before_jump':[30, 30, 30, 120, 50], 
          'transition_parameters': [0.1, 0.1], # the first value is its start value the second is its incriment 
          'transition':[True, True, True, False, False], 
          'learning_rate_generator':1.,
          'learning_rate_critic':1., 
          'generator_betas':(0.0, 0.99), 
          'critic_betas':(0.0, 0.99), 
          'ncrit':1, 
          'critic_lambda':10.,
          'epsilon_drift':0.001,
          'dataset_dir':'/home/deniz/Desktop/data_set/CelebAMask-HQ/', 
          'stat_format':'epoch {:4d} resolution {:4d} critic_loss {:6.4f} generator_loss {:6.4f} time {:6f}', 
          'dataset_address':'/home/deniz/Desktop/data_set/CelebAMask-HQ/'}

device = torch.device('cuda:0')


with open('stats.txt', 'a') as fid:
    fid.write('new simulation\n\n')


assert config['start_sr'] in config['sr'], "{} does not exist".format(config['start_sr'])
assert len(config['channels']) == len(config['sr']) + 1, 'channels must start from latent variable size as the first argument'

start_level = int(log2(config['start_sr']) - 2)

for level_index in range(start_level, len(config['sr'])):

    dataset_path = '{}CelebA_{}/'.format(config['dataset_dir'], config['sr'][level_index])
    transformation = Compose([ToTensor()])
    dataset = ImageFolder('{}/CelebA_{}/'.format(config['dataset_address'], config['sr'][level_index]), transform=transformation)
    dataloader = DataLoader(dataset, config['level_batch_size'][level_index], shuffle=True, num_workers=3)

    g_loss = [] 
    c_loss = []

    
    transition = True
    transition_coef = config['transition_parameters'][0] if config['transition'][level_index] else 2.

    for epoch in range(config['epochs_before_jump'][level_index]):

        # transition linearly 
        transition_coef += config['transition_parameters'][1]
        lr_gen = config['learning_rate_generator'] # / config['sr'][level_index] 
        lr_dis = config['learning_rate_critic']  # / config['sr'][level_index]

        if transition_coef <= 1. and transition:

            generator = Generator(config['sr'][level_index], config, transition=transition, transition_coef=transition_coef).to(device)
            critic = Discriminator(config['sr'][level_index], config, transition=transition, transition_coef=transition_coef).to(device)

            generator_optimizer = torch.optim.Adam(generator.parameters(), lr=lr_gen, betas=config['generator_betas'])
            critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr_dis, betas=config['critic_betas'])
        
        elif transition_coef >= 1. and transition:

            transition = False
            generator = Generator(config['sr'][level_index], config, transition=transition, transition_coef=transition_coef).to(device)
            critic = Discriminator(config['sr'][level_index], config, transition=transition, transition_coef=transition_coef).to(device)

            generator_optimizer = torch.optim.Adam(generator.parameters(), lr=lr_gen, betas=config['generator_betas'])
            critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr_dis, betas=config['critic_betas'])
        
        start_time = time()

        dataloader = DataLoader(dataset, config['level_batch_size'][level_index], shuffle=True, num_workers=3)
        ticker = 0
        j = 0
        for image, h in dataloader:
            j += 1

            data_distribution = (image * 255.).to(device)
            latent_variable = torch.randn(data_distribution.shape[0], config['latent_size']).to(device)

            # update critic
            critic_optimizer.zero_grad()
            ticker += 1
            model_distribution = generator(latent_variable)
            critic_loss = critic(model_distribution).mean() - critic(data_distribution).mean()
            # critic_loss += (critic(model_distribution) **2.).mean() * config['epsilon_drift']
            critic_loss += (critic(data_distribution) **2. ).mean() * config['epsilon_drift']
            
            critic_loss.backward()

            PG_Gradient_Penalty(data_distribution, model_distribution, critic, config['critic_lambda'], True)
            
            critic_optimizer.step()
            c_loss.append(critic_loss.tolist())
            
            if ticker < config['ncrit']:
                continue

            # update generator
            latent_variable = torch.randn(config['level_batch_size'][level_index], config['latent_size']).to(device)
            generator_optimizer.zero_grad()
            ticker = 0
            var = generator(latent_variable)
            generator_loss = -critic(var).mean()
            generator_loss.backward()
            generator_optimizer.step()
            g_loss.append(generator_loss.tolist())

            if j % 64 == 0:
                print(array(c_loss).mean(), -array(g_loss).mean())
                generator.save()
                critic.save()

        generator.save()
        critic.save()
        stat = config['stat_format'].format(epoch, config['sr'][level_index], array(c_loss).mean(), -array(g_loss).mean(), time() - start_time)
        with open('stats.txt', 'a') as fid:
            fid.write(stat + '\n')
        print(stat)
        d_loss, c_loss = [], []

'''
# test saving and loading

level_index = 1
# x = (torch.zeros(4, 3, config['sr'][level_index], config['sr'][level_index]) + 0.2).to(device)
x = torch.randn(4, 128).to(device)
transition = True

generator = Generator(config['sr'][level_index], config, transition=transition).to(device)
critic = Discriminator(config['sr'][level_index], config, transition=transition).to(device)
# print(generator(x))
# generator(x)
print(critic(generator(x)))

# print()

generator = Generator(config['sr'][level_index], config, transition=transition).to(device)
critic = Discriminator(config['sr'][level_index], config, transition=transition).to(device)
print(critic(generator(x)))
# generator(x)
# print(generator(x))
# print(critic(x))
'''