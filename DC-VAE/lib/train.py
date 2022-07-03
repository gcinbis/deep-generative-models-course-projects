from .models import Model
from .loss import contrastive_loss
from .val import eval
from .utils import *

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from tqdm.auto import tqdm
import os.path as osp
import wandb

def train(model_params, hparams, _run, checkpoint = None):
    
    device = hparams['device']

    torch.manual_seed(123)

    if device.type == "cuda":
        torch.cuda.manual_seed(123)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model = Model(model_params).to(device)
    
    model.apply(weights_init)

    configs = {'model_params': model_params, 'hparams': hparams}
    
    wandb.init(project="dcvae", entity="aybora", config = configs)

    wandb.define_metric("step")
    wandb.define_metric("epoch")
    wandb.define_metric("gen_loss_train", step_metric="step")
    wandb.define_metric("disc_loss_train", step_metric="step")
    wandb.define_metric("cont_loss_train", step_metric="step")
    wandb.define_metric("fid_samp", step_metric="epoch")
    wandb.define_metric("fid_recon", step_metric="epoch")

    if checkpoint is not None:
        print("Checkpoint is loaded !!!")
        model.load_state_dict(checkpoint)

    enc_optim = torch.optim.Adam(model.encoder.parameters(), lr = hparams['lr'], betas = (hparams['beta1'], hparams['beta2']))
    dec_optim = torch.optim.Adam(model.decoder.parameters(), lr = hparams['lr'], betas = (hparams['beta1'], hparams['beta2']))
    disc_optim = torch.optim.Adam(model.discriminator.parameters(), lr = hparams['lr'], betas = (hparams['beta1'], hparams['beta2']))

    gan_criterion = torch.nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        torchvision.datasets.CIFAR10(
            './data', 
            train = True,
            download = True, 
            transform = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                    (0.5, 0.5, 0.5), 
                                    (0.5, 0.5, 0.5)),
                           ])
            ),
        batch_size=hparams['train_batch_size'], 
        shuffle=True, 
        pin_memory=True,
        drop_last=True
        )
    
    test_loader = DataLoader(torchvision.datasets.CIFAR10(
        './data', 
        train = False, transform = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                    (0.5, 0.5, 0.5), 
                                    (0.5, 0.5, 0.5)),
                           ])
        ),
        batch_size=hparams['test_batch_size'], 
        shuffle=True,
        pin_memory=True, 
        drop_last=True
        )

    ### LOG ###
    disc_train_loss = 0
    gen_train_loss = 0
    cont_train_loss = 0

    mean_generator_loss = 0
    mean_discriminator_loss = 0
    mean_contrastive_loss = 0

    _run.info["gen_loss_train"] = list()
    _run.info["disc_loss_train"] = list()
    _run.info["cont_loss_train"] = list()
    _run.info["fid sampling"] = list()
    _run.info["fid recon"] = list()
    ##########

    disp_freq = hparams['disp_freq']
    step = 1
    
    for epoch in range(1,hparams['epochs']+1):
        
        iterator = tqdm(train_loader, leave=True)
        iterator.set_description_str(f"Epoch: {epoch}")
        batch_id = 0
        for point_batch, _ in iterator: 

            batch_id += 1
            
            model.train()
            model.device = device

            #### Real Data
            real_data = point_batch.to(device) 

            '''----------------         Discriminator Update         ----------------'''
            disc_optim.zero_grad()
            

            fake_data = model.gen_from_noise(size=(real_data.size(0), model_params['decoder']['latent_dim']))
            z_latent, rec_data = model(real_data)

            disc_fake_pred, _ = model.discriminator(fake_data)
            disc_fake_loss = gan_criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))

            disc_rec_pred, _ = model.discriminator(rec_data)
            disc_rec_loss = gan_criterion(disc_rec_pred, torch.zeros_like(disc_rec_pred))

            disc_real_pred, _ = model.discriminator(real_data)
            disc_real_loss = gan_criterion(disc_real_pred, torch.ones_like(disc_real_pred))

            gan_objective = disc_real_loss + (disc_rec_loss + disc_fake_loss)*0.5 
            gan_objective.backward(retain_graph = True)
            disc_optim.step()

            # Log
            _run.info["disc_loss_train"].append(gan_objective.item())
            if step % 10 == 0:
                wandb.log({"disc_loss_train": gan_objective.item(), "step": step})
            disc_train_loss = gan_objective.item()
            mean_discriminator_loss += gan_objective.item() 

            '''----------------         Generator Update         ----------------'''
            if step % hparams['gen_train_freq'] == 0:

                # KLD loss term missing !!!!!
                enc_optim.zero_grad()
                dec_optim.zero_grad()
                
                fake_data = model.gen_from_noise(size=(2*real_data.size(0), model_params['decoder']['latent_dim']))
                z_latent, rec_data = model(real_data)

                gen_fake_pred, _ = model.discriminator(fake_data)
                gen_fake_loss = gan_criterion(gen_fake_pred, torch.ones_like(gen_fake_pred))

                gen_rec_pred, _ = model.discriminator(rec_data)
                gen_rec_loss = gan_criterion(gen_rec_pred, torch.ones_like(gen_rec_pred))

                gan_objective =  gen_rec_loss + gen_fake_loss 

                gan_objective.backward(retain_graph = True)
                enc_optim.step()
                dec_optim.step()

                # Log
                _run.info["gen_loss_train"].append(gan_objective.item())
                if step % 10 == 0:
                    wandb.log({"gen_loss_train": gan_objective.item(), "step": step})
                gen_train_loss = gan_objective.item()
                mean_generator_loss += gan_objective.item() 
            
                '''----------------         Contrastive Update         ----------------'''

                enc_optim.zero_grad()
                dec_optim.zero_grad()
                disc_optim.zero_grad()

                z_latent, rec_data = model(real_data)

                _, rec_contrastive = model.discriminator(rec_data)
                _, real_contrastive = model.discriminator(real_data)

                cont_loss = contrastive_loss(z_latent, real_contrastive, rec_contrastive)

                cont_loss.backward()
                
                disc_optim.step()
                enc_optim.step()
                dec_optim.step()

                # Log
                _run.info["cont_loss_train"].append(cont_loss.item())
                if step % 10 == 0:
                    wandb.log({"cont_loss_train": cont_loss.item(), "step": step})
                cont_train_loss = cont_loss.item()
                mean_contrastive_loss += cont_loss.item() 
            
            # Visualize the generated images
            if step % disp_freq == 0:
                gen_images = model.gen_from_noise(size = (25, model_params['decoder']['latent_dim']))
                t_data, _ = iter(test_loader).next()
                t_data = t_data.to(device)
                _ , rec_t_data = model(t_data)
                show_img(gen_images, step, num_images=25, size=(3, 32, 32), img_save_path=osp.join(_run.experiment_info['base_dir'], 'runs', _run._id, 'results'), show=False, wandb_save = True)
                show_img_rec(t_data, rec_t_data, step, num_images=15, size=(3, 32, 32), img_save_path=osp.join(_run.experiment_info['base_dir'], 'runs', _run._id, 'results'), show=False, wandb_save = True)
            
            step += 1

            iterator.set_postfix_str(
                f"Disc Loss: {disc_train_loss:.4f}, Gen Loss: {gen_train_loss:.4f}, Cont Loss: {cont_train_loss:.4f}, Step: {step} " 
                )
            
            if step % hparams['checkpoint'] == 0:
                c_name = f"checkpoint_{step}.pt"
                checkpoint_path = osp.join(_run.experiment_info['base_dir'], 'runs', _run._id, "checkpoints", c_name)
                torch.save(model.state_dict(), checkpoint_path)
        
        if epoch % hparams['fid_freq'] == 0:
            fid_samp, fid_rec = eval(model, model_params['decoder']['latent_dim'], hparams['test_batch_size'], device, test_loader)
            print(f"Epoch: {epoch}| sampling fid: {fid_samp}| reconstruction fid: {fid_rec}")
            _run.info["fid sampling"].append(fid_samp.item())
            _run.info["fid recon"].append(fid_rec.item())
            wandb.log({"fid_samp": fid_samp.item(), "epoch": epoch})
            wandb.log({"fid_recon": fid_rec.item(), "epoch": epoch})

