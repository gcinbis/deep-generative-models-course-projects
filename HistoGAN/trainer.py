import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch_optimizer import DiffGrad

from loss import pl_reg, r1_reg, hellinger_dist_loss
from utils import truncation_trick, mixing_noise, random_interpolate_hists
from data import AnimeFacesDataset
from model import Discriminator, HistoGAN

class Trainer():
    def __init__(self, config):
        '''
        Initialize configuration parameters, dataset, dataloader, 
        generative and discriminator networks.
        '''
        training_dataset_path = config["training_dataset_path"]
        image_res = config["image_res"]
        transform = transforms.Compose(
                [transforms.Resize((image_res,image_res))])
        self.batch_size = config["batch_size"]
        self.num_epochs = config["num_epochs"]
        acc_gradient_total = config["acc_gradient_total"]
        self.acc_gradient_iter = acc_gradient_total //self.batch_size
        self.r1_factor = config["r1_factor"]
        self.r1_update_iter = config["r1_update_iter"]
        self.decay_coeff = config["decay_coeff"]
        self.plr_factor = np.log(2)/(256**2*(np.log(256)-np.log(2)))
        self.plr_update_iter = config["plr_update_iter"]
        self.save_iter = config["save_iter"]
        self.generated_images_path = config["generated_images_path"]
        if not os.path.isdir(self.generated_images_path):
            os.mkdir(self.generated_images_path)
        network_capacity = config["network_capacity"] 
        self.latent_dim = config["latent_dim"]
        self.bin_size = config["bin_size"]
        learning_rate = config["learning_rate"]
        mapping_layer_num = config["mapping_layer_num"]
        self.mixing_prob = config["mixing_prob"]
        self.num_gen_layers = int(np.log2(image_res)-1)
        self.use_plr = config["use_plr"]
        self.use_r1r = config["use_r1r"]
        kaiming_init= config["use_r1r"]
        use_eqlr = config["use_eqlr"]
        use_spec_norm = config["use_spec_norm"]
        self.loss_type = config["loss_type"]
        optim = config["optim"]
        optim_paras = config["optim_params"]
        self.device = config["device"]
        self.save_model_path = config["save_model_path"]
        pre_gen_path = config["pre_gen_path"]
        pre_disc_path = config["pre_disc_path"]
        gen_arch = config["gen_arch"]

        dataset = AnimeFacesDataset(training_dataset_path, transform, self.device)
        print("Dataset Loaded")
        self.dataloader = DataLoader(dataset, batch_size=acc_gradient_total, shuffle=True, drop_last=True)
        generator = HistoGAN(network_capacity, self.latent_dim, self.bin_size, image_res, mapping_layer_num, kaiming_init=kaiming_init, use_eqlr=use_eqlr, ver=gen_arch)
        discriminator = Discriminator(network_capacity, image_res, kaiming_init=kaiming_init, use_spec_norm=use_spec_norm)
        
        # If a pretrained network exists, load their parameters to continue training
        if not pre_gen_path is None and os.path.exists(pre_gen_path):
            generator.load_state_dict(torch.load(pre_gen_path))
            print("Pretrained weight for generator loaded from {}".format(pre_gen_path))
        if not pre_disc_path is None and os.path.exists(pre_disc_path):
            discriminator.load_state_dict(torch.load(pre_disc_path))
            print("Pretrained weight for discriminator loaded from {}".format(pre_disc_path))

        self.discriminator = discriminator.to(self.device)
        self.generator = generator.to(self.device)
        self.target_scale = torch.tensor([0], requires_grad=False).to(self.device)
        print("Networks Created")

        if optim == "Adam":
            gene_optim = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=optim_paras)
            disc_optim = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=optim_paras)
        elif optim == "DiffGrad":
            gene_optim = DiffGrad(generator.parameters(), lr=learning_rate, betas=optim_paras)
            disc_optim = DiffGrad(discriminator.parameters(), lr=learning_rate, betas=optim_paras)
        else:
            raise "Not Supported Optimizer Type!"
        
        self.gene_optim = gene_optim
        self.disc_optim = disc_optim

    def train_generator(self, hist_list, iter): 
        self.gene_optim.zero_grad()
        for target_hist in hist_list:
            # generate fake data using mixed noise
            z = mixing_noise(self.batch_size, self.num_gen_layers-2, self.latent_dim, self.mixing_prob).to(self.device)
            fake_data, w = self.generator(z, target_hist) 
            disc_score = self.discriminator(fake_data)
            
            # compute generator loss
            if self.loss_type in ["wasser", "hinge"]:
                g_loss = -torch.mean(disc_score)
            elif self.loss_type == "softplus":
                g_loss = torch.mean(torch.nn.functional.softplus(-disc_score))

            # compute histogram loss
            c_loss = hellinger_dist_loss(fake_data, target_hist, self.device)
            alpha = 2.0  # See Sec. 5.2 Training details
            g_loss += alpha * c_loss

            # compute path length loss
            if self.use_plr and (iter+1) % self.plr_update_iter == 0:
                pl_loss, self.target_scale = pl_reg(fake_data, w, self.target_scale, self.plr_factor, self.decay_coeff, self.device) 
                g_loss += pl_loss

            g_loss /= self.acc_gradient_iter
            g_loss.backward()

        self.gene_optim.step()
        self.gene_optim.zero_grad()

    def train_discriminator(self, chunk_data, iter):
        self.disc_optim.zero_grad()
        batch_size = self.batch_size
        hist_list = []
        for index in range(chunk_data.size(0)//batch_size):
            batch_data = chunk_data[index*batch_size:(index+1)*batch_size]
            batch_data.requires_grad_()
            batch_data = batch_data.to(self.device)
            target_hist = random_interpolate_hists(batch_data)
            hist_list.append(target_hist.clone())
            z = mixing_noise(self.batch_size, self.num_gen_layers-2, self.latent_dim, self.mixing_prob).to(self.device) 
            fake_data, _ = self.generator(z, target_hist) 
            fake_data = fake_data.detach()
            fake_scores = self.discriminator(fake_data)
            real_scores = self.discriminator(batch_data)
            if self.loss_type == "hinge":
                real_loss = torch.mean(torch.nn.functional.relu(1-real_scores))    
                fake_loss = torch.mean(torch.nn.functional.relu(1+ fake_scores)) 
            elif self.loss_type == "softplus":
                real_loss = torch.mean(torch.nn.functional.softplus(-real_scores))
                fake_loss = torch.mean(torch.nn.functional.softplus(fake_scores))
            elif self.loss_type == "wasser":
                real_loss = -torch.mean(real_scores)  
                fake_loss = torch.mean(fake_scores)

            if self.use_r1r and iter % self.r1_update_iter == 0:
                r1_loss =  r1_reg(batch_data, real_scores, self.r1_factor)
                real_loss += r1_loss
            
            real_loss /= self.acc_gradient_iter
            fake_loss /= self.acc_gradient_iter

            real_loss.backward()
            fake_loss.backward()

        self.disc_optim.step()
        self.disc_optim.zero_grad()
        return hist_list
