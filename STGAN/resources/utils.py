from matplotlib import pyplot as plt
import random
import os
import numpy as np
from torchvision.utils import make_grid, save_image
import torch


def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def generate_image_samples(images, gen, attrs_list, device='cpu'):

    images = images.to(device)

    org_attr = attrs_list[0]

    generated_image_list = [images]

    for attr in attrs_list:

        attr_diff = attr.to(device) - org_attr.to(device)
        attr_diff = attr_diff.to(device)

        generated_image = gen(images, attr_diff)
        generated_image_list.append(generated_image)
    
    # Stack generated images side to side
    generated_image_list = torch.cat(generated_image_list, dim=3)

    return generated_image_list

def show_images(image, device='cpu', sample_size=7, title="Images"):
    
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title(title)
    plt.imshow(np.transpose(make_grid(image.to(device)[:sample_size], padding=2, normalize=True).cpu(), (1, 2, 0)))


def save_images(image, image_name, sample_size=7):

    image = image.data.cpu()
    # image = (image + 1) / 2
    # image.clamp_(0, 1)

    save_image(image[:sample_size, :], image_name, normalize=True, nrow=1, padding=0)

def save_state(model, optimizer, epoch, model_name, path="./"):
    
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    file_name = f'{model_name}.pth'

    save_path = os.path.join(path, file_name)

    torch.save(state, save_path)


def load_state(model, optimizer, path, mode='train', device='cpu'):
    
    state = torch.load(path)

    model.load_state_dict(state['model_state_dict'])
    model.to(device)

    if mode == 'train':
        optimizer.load_state_dict(state['optimizer_state_dict'])
        model.train()
    else:
        model.train()

    epoch = state['epoch']