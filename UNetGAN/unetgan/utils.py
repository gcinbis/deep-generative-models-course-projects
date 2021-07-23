import torch
from tqdm import tqdm
import pathlib
from PIL import Image

from torchvision.utils import make_grid
import torchvision.transforms as transforms
from collections import OrderedDict

import os
from torch.utils.tensorboard import SummaryWriter

from .data import CelebA

def freeze(model):
    for p in model.parameters():
        p.requires_grad = False

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad = True

def make_deterministic(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Recursively remove using only pathlib: https://stackoverflow.com/a/58183834
def rm_tree(pth):
    pth = pathlib.Path(pth)
    for child in pth.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()

def get_cutmix_mask(img_size):
    """
    https://arxiv.org/pdf/1905.04899.pdf
    """
    w, h = img_size

    lamb = torch.rand(1)[0]
    cut_ratio = torch.sqrt(1 - lamb)

    r_x = torch.randint(w, (1,))[0]
    r_y = torch.randint(h, (1,))[0]
    r_w = (cut_ratio * w).type(torch.int)
    r_h = (cut_ratio * h).type(torch.int)

    x_1 = torch.clamp((r_x - r_w) // 2, 0, w)
    x_2 = torch.clamp((r_x + r_w) // 2, 0, w)
    y_1 = torch.clamp((r_y - r_h) // 2, 0, h)
    y_2 = torch.clamp((r_y + r_h) // 2, 0, h)

    mask = torch.ones(img_size)
    mask[x_1:x_2, y_1:y_2] = 0

    lamb = 1 - ((x_2 - x_1) * (y_2 - y_1)).float() / (w*h)

    return mask, lamb

def preview_samples(axis, samples, figure_size=None):
    """Show a batch of sampled images.
    Args:
        axis:       A matplotlib axis object
        samples:    A batch of tensors having shape (batch_size, 3, H, W)
    """
    if figure_size:
        axis.figure(figsize = figure_size)
    axis.imshow(make_grid(samples / 2.0 + 0.5).permute(1, 2, 0).cpu().numpy())


def create_sample_directory(model, sample_dir, device='cpu', num_samples=10000, overwrite=False):
    """Creates a directory of randomly sampled images. Probably used to create
    a fake dataset to compute FID.
    Note:
        The FID authors suggest to use a minimum sample size of 10000 to not
        underestimate the FID value.
    Args:
        model:          The generator model
        sample_dir:     Target directory to create images
        num_samples:    Number of samples to create
        device:         CPU or CUDA device object
        overwrite:      If True, then overwrite sample directory if it already exists
    """
    sample_path = pathlib.Path(sample_dir)

    path_exists = sample_path.exists()
    if path_exists:
        if overwrite:
            # Careful!
            rm_tree(sample_path)
        else:
            raise FileExistsError('Directory already exists, not overwriting!')
    sample_path.mkdir(parents=True)
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(num_samples)):
            # sample = model.sample(device=device, num_samples=1).squeeze(0)
            zs = torch.randn(size=(1, model.latent_dim), device=device)
            sample = model(zs).squeeze(0)
            # Unnormalize the image
            sample = 0.5 * sample + 0.5
            image = transforms.ToPILImage()(sample)
            image.save(str(sample_path / f'img{i}.png'), format='png')

def create_celeba_sample_directory(celeba_root_dir, sample_dir, num_samples=10000, overwrite=False, image_size=(128, 128)):
    """Selects and prepares randomly sampled CelebA images.
    
    Note:
        The FID authors suggest to use a minimum sample size of 10000 to not
        underestimate the FID value.
    Args:
        celeba_root_dir:    Directory of the root 'celeba' directory
        sample_dir:         Target directory to create images
        num_samples:        Number of samples to create
        device:             CPU or CUDA device object
        overwrite:          If True, then overwrite sample directory if it already exists
        image_size:         Saved image size 
    """
    sample_path = pathlib.Path(sample_dir)

    path_exists = sample_path.exists()
    if path_exists:
        if overwrite:
            # Careful!
            rm_tree(sample_path)
        else:
            raise FileExistsError('Directory already exists, not overwriting!')
    sample_path.mkdir(parents=True)

    dataset = CelebA(root=celeba_root_dir, split='all', download=False, transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    dataloader_iter = iter(dataloader)

    for i in tqdm(range(num_samples)):
        try:
            sample = next(dataloader_iter)
        except StopIteration:
            # Re-roll the iterator
            dataloader_iter = iter(dataloader)
            sample = next(dataloader_iter)
        sample = sample[0].squeeze(0)
        # Unnormalize
        sample = 0.5 * sample + 0.5
        image = transforms.ToPILImage()(sample)
        image.save(str(sample_path / f'img{i}.png'), format='png')


class Logger(object):
    def __init__(self, log_dir, log_path):
        """Create a SummaryWriter log to log_dir."""
        log_dir = os.path.join(log_dir, log_path)
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """Log multiple scalar variable."""
        for tag, value in tag_value_pairs:
            self.writer.add_scalar(tag, value, step)

def load_pretrained_models(d, g, pth):
    """Loads state dicts from a saved pth file.
    See PyTorch forums on why we need to remove the 'module.' prefix.
    """
    weight_dict = torch.load(pth)
    # Load D
    new_dis_state_dict = OrderedDict()
    for k, v in weight_dict['discriminator'].items():
        name = k[7:] # remove `module.`
        new_dis_state_dict[name] = v
    # Load G
    new_gen_state_dict = OrderedDict()
    for k, v in weight_dict['generator'].items():
        name = k[7:] # remove `module.`
        new_gen_state_dict[name] = v
    # load params
    print(d.load_state_dict(new_dis_state_dict))
    print(g.load_state_dict(new_gen_state_dict))