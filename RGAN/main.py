import argparse
from torch.utils.data import DataLoader
import torch
from utils import get_model, get_loss, get_dataset, cycle, grad_penalty, sample_fid, is_negative
import os
import numpy as np
import time
import random

parser = argparse.ArgumentParser()

parser.add_argument('--total_iter', type=int, default=100000, help='total number of iterations to train the generator')
parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
parser.add_argument('--seed', type=int, default=1, help='default seed for torch, numpy and random; (set 0 to train with a random seed)')
parser.add_argument('--d_iter', type=int, default=1, help='the number of iterations to train the discriminator before training the generator')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset type, "cifar10" or "cat"')
parser.add_argument('--model', type=str, default='standard_cnn', help='model architecture, "standard_cnn" or "dcgan_64"')
parser.add_argument('--loss_type', type=str, default='sgan', help='loss type, "sgan", "rsgan", "rasgan", "lsgan", "ralsgan", "hingegan", "rahingegan", "wgan-gp", "rsgan-gp" or "rasgan-gp"')
parser.add_argument('--cuda', type=bool, default=True, help = 'gpu training')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate for the discriminator and the generator')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 value of adam')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 value of adam')
parser.add_argument('--spec_norm', type=bool, default=False, help = 'spectral normalization for the discriminator')
parser.add_argument('--no_BN', type=bool, default=False, help = 'do not use batchnorm for any of the models')
parser.add_argument('--all_tanh', type=bool, default=False, help = 'use tanh for all activations of the models')
parser.add_argument('--lambd', type=int, default=10, help='coefficient for gradient penalty')
parser.add_argument('--n_workers', type=int, default=4, help='number of cpu threads to be used in data loader')

parser.add_argument('--fid_iter', type=is_negative, default=100000, help='iteration frequency for generating samples using generator for FID (set 0 to not generate)')
parser.add_argument('--log_iter', type=is_negative, default=0, help = 'iteration frequency for updating the log file (set 0 to not create a log file)')
parser.add_argument('--print_iter', type=is_negative, default=1000, help = 'iteration frequency for printing losses and time passed (set 0 to not print)')
parser.add_argument('--save_model', type=is_negative, default=100000, help = 'iteration frequency for saving the models (set 0 to not save)')

args = parser.parse_args()



# set args.device to be used for creating tensors
if args.cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')

elif args.cuda:
    print("GPU training is selected but it is not available, CPU will be used instead.")
    args.device = torch.device('cpu')
    args.cuda = False

else:
    args.device = torch.device('cpu')


# create the datasets folder
os.makedirs("datasets", exist_ok=True)

# create folders if needed
if(args.log_iter):
    os.makedirs("losses", exist_ok=True)
    losses = open(f"losses/{args.dataset}_{args.loss_type}_n_d_{args.d_iter}_b1_{args.beta1}_b2_{args.beta2}_b_size_{args.batch_size}_lr_{args.lr}" +  ( "_noBN" if args.no_BN else "") + ("_alltanh" if args.all_tanh else "") + ".txt", "a+")

if(args.save_model):
    os.makedirs("models", exist_ok=True)

if(args.fid_iter):
    os.makedirs("samples", exist_ok=True)



if(args.seed):  # use a pre-determined seed for numpy, random and torch if args.seed is set
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if(args.cuda):
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True



#create the models
Generator, Discriminator = get_model(args)


# optimizers for the generator and discriminator
optimizer_G = torch.optim.Adam(params=Generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
optimizer_D = torch.optim.Adam(params=Discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))


# get loss functions according to the given loss type
gen_loss, disc_loss = get_loss(args.loss_type)


# booleans for gradient penalty, relativistic loss and relativistic average loss. used for determining the loss parameters
gradient_pen = args.loss_type in ["wgan-gp", "rsgan-gp", "rasgan-gp"]
relativistic = args.loss_type in ["rsgan", "rasgan", "ralsgan", "rahingegan", "rsgan-gp","rasgan-gp"]
average = args.loss_type in ["rasgan", "ralsgan", "rahingegan", "rasgan-gp"]




# get dataset (cifar10 or cat)
dataset = get_dataset(args.dataset)


if(args.fid_iter):  # setting the number of samples that is going to be generated to the number of images in the training dataset
    args.fid_sample = len(dataset)    


dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size ,shuffle=True,  num_workers=args.n_workers)

# an iterable object that resets the loader when it is completely iterated
loader_iter = iter(cycle(dataloader, args.dataset, args.device)) 


if(args.print_iter):    # start timers for total time and print time
    print_time = time.time()
    start_time = time.time()


for i in range(args.total_iter):

    for _ in range(args.d_iter):

        optimizer_D.zero_grad()

        real = next(loader_iter)

        noise = torch.randn(size=(real.size(0),128,1,1), device=args.device)

        fake = Generator(noise).detach()

        loss_args_D = [Discriminator(real), Discriminator(fake)] # default discriminator loss parameters for every loss type
        

        if(average):  # averages of the discriminator outputs if the loss type is relativistic average
            
            loss_args_D += [loss_args_D[0].mean(), loss_args_D[1].mean()]  


        if(gradient_pen): # create an interpolated input and calculate the gradient penalty

            random_num = torch.rand(size=(real.size(0),1,1,1), device=args.device)

            x_hat = random_num * real + (1-random_num) * fake

            loss_args_D += [grad_penalty(Discriminator, x_hat, args.lambd)]

        loss_D = disc_loss( *loss_args_D )

        loss_D.backward()

        optimizer_D.step()


    optimizer_G.zero_grad()

    noise = torch.randn(size=(real.size(0),128,1,1), device=args.device)    

    fake = Generator(noise)
    
    
    loss_args_G = [Discriminator(fake)]    # default generator loss parameter for every loss type


    if(relativistic):  # relativistic loss uses discriminator output for real images

        real = next(loader_iter)

        loss_args_G = [Discriminator(real)] + loss_args_G


    if(average):  # averages of the discriminator outputs if the loss type is relativistic average

        loss_args_G += [loss_args_G[0].mean(), loss_args_G[1].mean()]

    loss_G = gen_loss( *loss_args_G )

    loss_G.backward()

    optimizer_G.step()

    
    if(args.log_iter and (i+1) % args.log_iter == 0):    # update log file

        losses.write(f"{i}/{args.total_iter} loss_D {loss_D.item():.6f} loss_G {loss_G.item():.6f}\n")



    if(args.print_iter and (i+1) % args.print_iter == 0 ):   # print losses and the time passed since the last print

        print(f"Iter[{i+1}/{args.total_iter}] loss_D {loss_D.item():3f} loss_G {loss_G.item():.3f} {time.time()-print_time} s passed since the last print")

        print_time = time.time()

    
    if(args.save_model and (i+1) % args.save_model == 0 and (i+1) != 10000):

        torch.save(Generator.state_dict(), f"models/gen_{args.dataset}_{args.loss_type}_n_d_{args.d_iter}_b1_{args.beta1}_b2_{args.beta2}_b_size_{args.batch_size}_lr_{args.lr}_{i+1}" +  ( "_noBN" if args.no_BN else "") + ("_alltanh" if args.all_tanh else "") + ".pth")
        #torch.save(Discriminator.state_dict(), f"models/disc_{args.dataset}_{args.loss_type}_n_d_{args.d_iter}_b1_{args.beta1}_b2_{args.beta2}_b_size_{args.batch_size}_lr_{args.lr}_{i+1}" +  ( "_noBN" if args.no_BN else "") + ("_alltanh" if args.all_tanh else "") + ".pth")

        
    if(args.fid_iter and (i+1) % args.fid_iter == 0 and (i+1) != 10000):   # generate samples for calculating Frechet Inception Distance

        sampling_time = time.time()

        sample_fid(Generator, i, args)

        if(args.print_iter):
            print(f"Iter[{i+1}/{args.total_iter}] sampling took {time.time()-sampling_time} s ")



# end of the training

if(args.log_iter):

    losses.close()

if(args.print_iter):

    end_time = time.time()
    print(f"Total training time {(end_time-start_time)//60} minutes {((end_time-start_time)%60):.1f} seconds")
