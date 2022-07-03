import os
import numpy as np
import torch
from torchvision.utils import save_image
from trainer import Trainer





# Taken from https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html

import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.utils import make_grid


plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs):
    imgs = imgs.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8)
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.show()

config = dict(
    num_epochs = 10, # number of epochs for training
    batch_size = 16, # batch size
    acc_gradient_total = 16, # total number of samples seen by the networks in 1 iteration
    r1_factor = 10, # coefficient of the r1 regularization term
    r1_update_iter = 4, # in every r1_update_iter r1 regularization is used
    decay_coeff = 0.99, # ema decay coefficient for updating the path length target varaible
    plr_update_iter = 32, # in every plr_update_iter the path length regularization is used
    save_iter = 400, # in every save_iter the images are saved
    image_res = 64, # the resolution of the images
    network_capacity = 16, # capacity of the network used for channels of constant input in generator 
    latent_dim = 512, # dimensionalty of the noises
    bin_size = 64, # bin size of the histograms
    learning_rate = 0.0002, # learning rate
    mapping_layer_num = 8, # number of Linear layers in Mapping part of the Generator (z -> w)
    mixing_prob = 0.9, # probality of using two distinct noises for generation
    use_plr = True, # Wheter to use path length reg in training
    use_r1r = True, # Wheter to use r1 reg in training
    kaiming_init = True, # Initiazlize networks with kaiming initialization method by He et al.
    use_eqlr = False, # use eqularized learning coefficients for weights (similar to kaiming but used in every forward calculation)
    use_spec_norm = False, # use spectral normalization of Discriminator weights (For stabilization)
    disc_arch= "ResBlock", # architecture of the Discriminator (used for bookkeeping)
    gen_arch = "InputModDemod", # architecture of the Generator (used for bookkeeping)
    optim="Adam", # Optimizer used (Adam or DiffGrad)
    optim_params = (0.5, 0.9),  # Optimizer beta values (Adam and DiffGrad)
    loss_type="wasser", # Loss type to use (Wasserstein, Hinge, Log Sigmoid)
    save_model_path = ".", # Path to save generator and discriminator
    pre_gen_path = None, # for loading a pretrained network
    pre_disc_path = None, # for loading a pretrained network
    training_dataset_path = "images/anime_face", # path of training images 
    generated_images_path = "generated_images", # path to save generated images
    device="cuda" if torch.cuda.is_available() else "cpu"
    )

trainer = Trainer(config=config)


# Traning loop with gradient accumulation
total_iter = 0
for epoch in range(0, trainer.num_epochs):
    for iter, chunk_data in enumerate(trainer.dataloader):
        total_iter += 1
        hist_list = trainer.train_discriminator(chunk_data, iter)
        trainer.train_generator(hist_list, iter)

    # save iamges after every epoch
        if iter % trainer.save_iter == 0:
            z = torch.randn(trainer.batch_size, trainer.num_gen_layers,trainer.latent_dim).to(trainer.device)
            fake_data, _ = trainer.generator(z, hist_list[0])
            print(fake_data.size())
            save_image(fake_data, "{}.png".format(epoch), normalize=True)
            # fd = [data for data in fake_data]
            # grid = make_grid(fd, nrow=1, normalize=True)
            # show(grid)
            # plt.show()
            # del grid, fd, fake_data, z
            torch.save(trainer.generator.state_dict(), os.path.join(trainer.save_model_path, "generator.pt"))
            torch.save(trainer.discriminator.state_dict(), os.path.join(trainer.save_model_path, "discriminator.pt"))