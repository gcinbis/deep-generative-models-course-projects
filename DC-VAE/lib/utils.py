import torch
from torchvision.utils import make_grid
import torchvision
import matplotlib.pyplot as plt

def UnNormalize(tensor, mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)):

    if tensor.dim() == 3:
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor

    if tensor.dim() == 4:
        for idx in range(len(tensor)):
            ten = tensor[idx, : , :, :]
            for t, m, s in zip(ten, mean, std):
                t.mul_(s).add_(m)
        return tensor
    

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_uniform(m.weight.data, 1.)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)



def show_img(img, step = 0, num_images=25, size=(3, 32, 32), img_save_path = None, show = True, wandb_save=True):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    '''
    image_unflat = UnNormalize(img.clone().detach()).cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    if img_save_path is not None:
        torchvision.utils.save_image(image_grid, f"{img_save_path}/step_{step}.png")
    if show:
        plt.imshow(image_grid.permute(1, 2, 0).squeeze())
        plt.show()
    if wandb_save:
        import wandb
        images = wandb.Image(image_grid, caption = f"sampling of checkpoint {step}")
        wandb.log({"sampling examples": images})    
        

def show_img_rec(img, rec_img ,step = 0, num_images=15, size=(3, 32, 32), img_save_path = None, show = True, wandb_save = True):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    '''
    img_unflat = UnNormalize(img.clone().detach()).cpu().view(-1, *size)
    rec_img_unflat =UnNormalize(rec_img.clone().detach()).cpu().view(-1, *size)
    im = torch.cat([img_unflat[:num_images], rec_img_unflat[:num_images]], dim=0)

    image_grid = make_grid(im, nrow = 5) 
    if img_save_path is not None:
        torchvision.utils.save_image(image_grid, f"{img_save_path}/step_{step}_rec.png")

    if show:
        plt.imshow(image_grid.permute(1, 2, 0).squeeze())
        plt.show()

    if wandb_save:
        import wandb
        images = wandb.Image(image_grid, caption = f"recon of checkpoint {step}")
        wandb.log({"recon examples": images})  


