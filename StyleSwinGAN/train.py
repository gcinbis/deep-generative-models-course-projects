import torch
import torch.nn as nn
import torchvision 
import torchvision.datasets as datasets
import time
from dataset import get_sample
from utils import adjust_gradient, discriminator_loss, generator_loss, gradient_penalty

def train(loader, generator, discriminator, g_optim, d_optim, args):
    # Get Loader
    loader = get_sample(loader)

    # Gradient penalty loss initialization
    r1_loss = torch.tensor(0.0, device=args.device) 

    # Log the start of training
    print(" TRAINING HAS STARTED... ")
    
    end = time.time()
    
    # If decay_abrubt is true Generator learning rate is decated 1/4 of Discriminator learning rate
    # As described in the paper
    if args.decay_abrubt:
        G_lr = args.D_lr / 4
    
    # Decay learning rate with respect to iteration number
    if args.lr_decay:
        lr_decay_per_step = G_lr / (args.iters - args.lr_decay_start_steps)

    # Training iterations.
    for idx in range(args.iters):
        i = idx + args.start_iters
        if i > args.iters:
          break
        
        ################################## Train Discriminator ##################################

        generator.train()

        # Get the data
        this_data = next(loader) # This will load a 2-tuple
        real_img = this_data[0]  # Get the first component
        real_img = real_img.to(args.device) 

        # Freeze Generator and set Discriminator Trainable
        adjust_gradient(generator, False)
        adjust_gradient(discriminator, True)

        # Sample random noise
        noise = torch.randn((args.batch, args.style_dim)).cuda()

        # Generate fake batch of image
        fake_img = generator(noise)

        # Calculate discriminator loss
        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img)
        d_loss = discriminator_loss(real_pred, fake_pred) * args.gan_weight

        # Update parameters
        discriminator.zero_grad()
        d_loss.backward()
        nn.utils.clip_grad_norm_(discriminator.parameters(), 5.0)
        d_optim.step()

        # Regularize with gradient penalyzation - R1 Loss
        d_regularize = i % args.d_reg_every == 0
        if d_regularize:
            real_img.requires_grad = True
            # Get the predictions
            real_pred = discriminator(real_img)
            # Calculate R1 Loss
            r1_loss = gradient_penalty(real_pred, real_img)
            # Update parameters
            discriminator.zero_grad()
            (args.gan_weight * (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0])).backward()
            d_optim.step()

        ################################## Train Generator  ##################################
        # Freeze Discriminator, set Generator trainable
        adjust_gradient(generator, True)
        adjust_gradient(discriminator, False)

        # Get the data
        this_data = next(loader) # This will load a 2-tuple
        real_img = this_data[0]  # Get the first component
        real_img = real_img.to(args.device)

        # Sample random noise
        noise = torch.randn((args.batch, args.style_dim)).cuda()
        
        # Generate fake batch of image
        fake_img = generator(noise)
        
        # Calculate generator loss
        fake_pred = discriminator(fake_img)
        g_loss = generator_loss(fake_pred)* args.gan_weight

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        # Log and Save
        if i % args.print_freq == 0:
            iters_time = time.time() - end
            end = time.time()
            
            print("Iters: {}\tTime: {:.4f}\tD_loss: {:.4f}\tG_loss: {:.4f}\tR1: {:.4f}".format(i, iters_time, d_loss, g_loss, args.r1))

        if i != 0 and i % args.save_freq ==0:    
          # Experiment name
          exp = args.datasetname + '_' +  '_' + str(args.expname) + '_' + str(i)
          # Save the checkpoint
          torch.save({'generator': generator.state_dict(),
                      'discriminator': discriminator.state_dict(),
                      'args': args}, args.save_path + 'state_dict_' + exp)