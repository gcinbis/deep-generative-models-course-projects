"""
HoloGAN implementation in PyTorch
May 17, 2020
"""
import os
import csv
import time
import math
import collections
import torch
import numpy as np
#import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join
from torch import nn
from torch.optim import Adam
from torch.autograd import Variable
from torchvision import datasets, transforms
from scipy.misc import imsave
from datetime import datetime
from discriminator import Discriminator
from generator import Generator

class HoloGAN():
    """HoloGAN.

    HoloGAN model is the Unsupervised learning of 3D representations from natural images.
    The paper can be found in https://www.monkeyoverflow.com/hologan-unsupervised-learning-\
    of-3d-representations-from-natural-images/
    """
    def __init__(self, args):
        super(HoloGAN, self).__init__()

        torch.manual_seed(args.seed)
        use_cuda = args.gpu and torch.cuda.is_available()
        args.device = torch.device("cuda" if use_cuda else "cpu")

        # model configurations
        if args.load_dis is None:
            self.discriminator = Discriminator(in_planes=3, out_planes=64,
                                               z_planes=args.z_dim).to(args.device)
        else:
            self.discriminator = torch.load(args.load_dis).to(args.device)

        if args.load_gen is None:
            self.generator = Generator(in_planes=64, out_planes=3,
                                       z_planes=args.z_dim, gpu=use_cuda).to(args.device)
        else:
            self.generator = torch.load(args.load_gen).to(args.device)

        # optimizer configurations
        self.optimizer_discriminator = Adam(self.discriminator.parameters(),
                                            lr=args.d_lr, betas=(args.beta1, args.beta2))
        self.optimizer_generator = Adam(self.generator.parameters(),
                                        lr=args.d_lr, betas=(args.beta1, args.beta2))

        # Load dataset
        self.train_loader = self.load_dataset(args)

        # create result folder
        args.results_dir = os.path.join("results", args.dataset)
        if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)

        # create history file
        args.hist_file = open(os.path.join(args.results_dir, "history.csv"), "a", newline="")
        args.recorder = csv.writer(args.hist_file, delimiter=",")
        if os.stat(os.path.join(args.results_dir, "history.csv")).st_size == 0:
            args.recorder.writerow(["epoch", "time", "d_loss", "g_loss", "q_loss"])

        # create model folder
        args.models_dir = os.path.join("models", args.dataset)
        if not os.path.exists(args.models_dir):
            os.makedirs(args.models_dir)

        # continue to broken training
        args.start_epoch = 0
        if args.load_dis is None:
            load_model = ""
            for modelname in listdir(args.models_dir):
                if isfile(join(args.models_dir, modelname)) and \
                   ("discriminator.v" in modelname or "generator.v" in modelname):
                    start_loc = modelname[:-3].rfind(".v") + 2
                    end_loc = modelname[:-3].rfind("_")
                    epoch_str = modelname[start_loc:end_loc]
                    batch_str = modelname[end_loc:]
                    dis_model = os.path.join(args.models_dir, "discriminator.v"+epoch_str+batch_str)
                    gen_model = os.path.join(args.models_dir, "generator.v"+epoch_str+batch_str)
                    if args.start_epoch < int(epoch_str) and os.path.exists(dis_model) and os.path.exists(gen_model):
                        args.start_epoch = int(epoch_str)
                        load_model = epoch_str + batch_str

            if args.start_epoch > 0:
                print("Broken training is detected. Starting epoch is", args.start_epoch)
                dis_model = os.path.join(args.models_dir, "discriminator.v"+load_model)
                gen_model = os.path.join(args.models_dir, "generator.v"+load_model)
                self.discriminator = torch.load(dis_model).to(args.device)
                self.generator = torch.load(gen_model).to(args.device)

        # create sampling folder
        args.samples_dir = os.path.join("samples", args.dataset)
        if not os.path.exists(args.samples_dir):
            os.makedirs(args.samples_dir)

    def train(self, args):
        """HoloGAN trainer

        This method train the HoloGAN model.
        """
        d_lr = args.d_lr
        g_lr = args.g_lr
        for epoch in range(args.start_epoch, args.max_epochs):
            # Adaptive learning rate
            if epoch >= args.epoch_step:
                adaptive_lr = (args.max_epochs - epoch) / (args.max_epochs - args.epoch_step)
                d_lr *= adaptive_lr
                g_lr *= adaptive_lr
                for param_group in self.optimizer_discriminator.param_groups:
                    param_group['lr'] = d_lr
                for param_group in self.optimizer_generator.param_groups:
                    param_group['lr'] = g_lr

            result = collections.OrderedDict({"epoch":epoch})
            result.update(self.train_epoch(args, epoch))
            # validate and keep history at each log interval
            self.save_history(args, result)

        # save the model giving the best validation results as a final model
        if not args.no_save_model:
            self.save_model(args, args.max_epochs-1, best=True)

    def train_epoch(self, args, epoch):
        """train an epoch

        This method train an epoch.
        """
        batch = {"time":[], "g":[], "d":[], "q":[]}
        self.generator.train()
        self.discriminator.train()
        original_batch_size = args.batch_size
        for idx, (data, _) in enumerate(self.train_loader):
            print("Epoch: [{:2d}] [{:3d}/{:3d}] ".format(epoch, idx, len(self.train_loader)), end="")
            x = data.to(args.device)
            args.batch_size = len(x)
            # rnd_state = np.random.RandomState(seed)
            z = self.sample_z(args)
            view_in = self.sample_view(args)

            d_loss, g_loss, q_loss, elapsed_time = self.train_batch(x, z, view_in, args, idx)
            batch["d"].append(float(d_loss))
            batch["g"].append(float(g_loss))
            batch["q"].append(float(q_loss))
            batch["time"].append(float(elapsed_time))

            # print the training results of batch
            print("time: {:.2f}sec, d_loss: {:.4f}, g_loss: {:.4f}, q_loss: {:.4f}"
                  .format(elapsed_time, float(d_loss), float(g_loss), float(q_loss)))

            if (idx % args.log_interval == 0):
                self.sample(args, epoch, idx, collection=True)
                # save model parameters
                if not args.no_save_model:
                    self.save_model(args, epoch, idx)

        result = {"time"  : round(np.mean(batch["time"])),
                  "d_loss": round(np.mean(batch["d"]), 4),
                  "g_loss": round(np.mean(batch["g"]), 4),
                  "q_loss": round(np.mean(batch["q"]), 4)}
        args.batch_size = original_batch_size
        return result

    def train_batch(self, x, z, view_in, args, batch_id):
        """train the given batch

        Arguments are
        * x:        images in the batch.
        * z:        latent variables in the batch.
        * view_in:  3D transformation parameters.

        This method train the given batch and return the resulting loss values.
        """
        start = time.process_time()
        loss = nn.BCEWithLogitsLoss()

        # Train the generator.
        self.optimizer_generator.zero_grad()
        fake = self.generator(z, view_in)
        d_fake, g_z_pred = self.discriminator(fake[:, :, :64, :64])
        one = torch.ones(d_fake.shape).to(args.device)
        gen_loss = loss(d_fake, one)
        q_loss = torch.mean((g_z_pred - z)**2)
        if batch_id % args.update_g_every_d == 0:
            (gen_loss + args.lambda_latent * q_loss).backward()
            self.optimizer_generator.step()

        # Train the discriminator.
        self.optimizer_discriminator.zero_grad()
        d_fake, d_z_pred = self.discriminator(fake[:, :, :64, :64].detach())
        d_real, _ = self.discriminator(x)
        one = torch.ones(d_real.shape).to(args.device)
        zero = torch.zeros(d_fake.shape).to(args.device)
        dis_loss = loss(d_real, one) + loss(d_fake, zero)
        q_loss = torch.mean((d_z_pred - z)**2)
        (dis_loss + args.lambda_latent * q_loss).backward()
        self.optimizer_discriminator.step()

        elapsed_time = time.process_time()  - start
        return float(dis_loss), float(gen_loss), float(q_loss), elapsed_time

    def sample(self, args, epoch=0, batch=0, trained=False, collection=False):
        """HoloGAN sampler

        This samples images in the given configuration from the HoloGAN.
        Images can be found in the "args.samples_dir" directory.
        """
        z = self.sample_z(args)
        if args.rotate_azimuth:
            low, high, step = args.azimuth_low, args.azimuth_high+1, 5
        elif args.rotate_elevation:
            low, high, step = args.elevation_low, args.elevation_high, 5
        else:
            low, high, step = 0, 10, 1

        if not trained:
            folder = os.path.join(args.samples_dir, "epoch"+str(epoch)+"_"+str(batch))
        else:
            now = datetime.now()
            timestamp = datetime.timestamp(now)
            folder = os.path.join(args.samples_dir, "sample_"+str(timestamp))

        if not os.path.exists(folder):
            os.makedirs(folder)

        for i in range(low, high, step):
            # Apply only azimuth rotation
            if args.rotate_azimuth:
                view_in = torch.tensor([i*math.pi/180, 0, 1.0, 0, 0, 0])
                view_in = view_in.repeat(args.batch_size, 1)
            # Apply only elevation rotation
            elif args.rotate_elevation:
                view_in = torch.tensor([270*math.pi/180, i*math.pi/180, 1.0, 0, 0, 0])
                view_in = view_in.repeat(args.batch_size, 1)
            # Apply default transformation
            else:
                view_in = self.sample_view(args)

            samples = self.generator(z, view_in).permute(0, 2, 3, 1)
            normalized = ((samples+1.)/2.).cpu().detach().numpy()
            image = np.clip(255*normalized, 0, 255).astype(np.uint8)

            if collection and args.batch_size >= 4:
                imsave(os.path.join(folder, "samples_"+str(i)+".png"),
                       self.merge_samples(image, [args.batch_size // 4, 4]))
            else:
                imsave(os.path.join(folder, "samples_"+str(i)+".png"), image[0])

            if trained:
                print("Samples are saved in", os.path.join(folder, "samples_"+str(i)+".png"))

    def load_dataset(self, args):
        """dataset loader.

        This loads the dataset.
        """
        kwargs = {'num_workers': 2, 'pin_memory': True} if args.device == 'cuda' else {}

        if args.dataset == 'celebA':
            transform = transforms.Compose([\
                transforms.CenterCrop(108),
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

            trainset = datasets.ImageFolder(root=args.image_path, transform=transform)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,\
                        shuffle=True, **kwargs)
        return train_loader

    def sample_z(self, args):
        """Latent variables sampler

        This samples latent variables from the uniform distribution [-1,1].
        """
        tensor = torch.cuda.FloatTensor if args.device == "cuda" else torch.FloatTensor
        size = (args.batch_size, args.z_dim)
        return tensor(np.random.uniform(-1., 1., size)).to(args.device)

    def sample_view(self, args):
        """Transformation parameters sampler

        This samples view (or transformation parameters) from the given configuration.
        """
        # the azimuth angle (theta) is around y
        theta = np.random.randint(args.azimuth_low, args.azimuth_high,
                                  (args.batch_size)).astype(np.float)
        theta = theta * math.pi / 180.0

        # the elevation angle (gamma) is around x
        if args.elevation_low < args.elevation_high:
            gamma = np.random.randint(args.elevation_low, args.elevation_high,
                                      (args.batch_size)).astype(np.float)
            gamma = gamma * math.pi / 180.0
        else:
            gamma = np.zeros(args.batch_size).astype(np.float)

        scale = float(np.random.uniform(args.scale_low, args.scale_high))
        shift_x = args.transX_low + np.random.random(args.batch_size) * \
                  (args.transX_high - args.transX_low)
        shift_y = args.transY_low + np.random.random(args.batch_size) * \
                  (args.transY_high - args.transY_low)
        shift_z = args.transZ_low + np.random.random(args.batch_size) * \
                  (args.transZ_high - args.transZ_low)

        view = np.zeros((args.batch_size, 6))
        column = np.arange(0, args.batch_size)
        view[column, 0] = theta
        view[column, 1] = gamma
        view[column, 2] = scale
        view[column, 3] = shift_x
        view[column, 4] = shift_y
        view[column, 5] = shift_z
        return view

    def save_history(self, args, record):
        """save a record to the history file"""
        args.recorder.writerow([str(record[key]) for key in record])
        args.hist_file.flush()

    def save_model(self, args, epoch, batch=0, best=False):
        """save model

        Arguments are
        * epoch:   epoch number.
        * best:    if the model is in the final epoch.

        This method saves the trained discriminator and generator in a pt file.
        """
        if best is False:
            dis_model = os.path.join(args.models_dir, "discriminator.v"+str(epoch)+"_"+str(batch)+".pt")
            gen_model = os.path.join(args.models_dir, "generator.v"+str(epoch)+"_"+str(batch)+".pt")
            torch.save(self.discriminator, dis_model)
            torch.save(self.generator, gen_model)
        else:
            batch = len(self.train_loader)-1
            dis_model = os.path.join(args.models_dir, "discriminator.v"+str(epoch)+"_"+str(batch)+".pt")
            gen_model = os.path.join(args.models_dir, "generator.v"+str(epoch)+"_"+str(batch)+".pt")
            while batch > 0 and not (os.path.exists(dis_model) and os.path.exists(gen_model)):
                batch -= 1
                dis_model = os.path.join(args.models_dir, "discriminator.v"+str(epoch)+"_"+str(batch)+".pt")
                gen_model = os.path.join(args.models_dir, "generator.v"+str(epoch)+"_"+str(batch)+".pt")

            train_files = os.listdir(args.models_dir)
            for train_file in train_files:
                if not train_file.endswith(".v"+str(epoch)+"_"+str(batch)+".pt"):
                    os.remove(os.path.join(args.models_dir, train_file))

            os.rename(dis_model, os.path.join(args.models_dir, "discriminator.pt"))
            os.rename(gen_model, os.path.join(args.models_dir, "generator.pt"))

    def merge_samples(self, images, size):
        _, h, w, c = images.shape
        collection = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            collection[j*h : j*h+h, i*w : i*w+w, :] = image
        return collection
