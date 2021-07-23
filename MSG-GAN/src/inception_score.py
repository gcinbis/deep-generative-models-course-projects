import torch
from torch.nn import Module
from torchvision.models.inception import inception_v3
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from torchvision.transforms.functional import resize, normalize
from math import ceil

class Inception_Score(Module):
    """
    Class to compute Inception Score published in 
    'Improved Techniques for Training GANs' (arxiv.org/abs/1606.03498).
    __init__ function downloads the pretrained inception_v3 model.
    forward function computes the score.
    """
    def __init__(self, device):
        super(Inception_Score, self).__init__()
        self.model = inception_v3(pretrained=True, transform_input=True).to(device)
        self.model.eval()
        self.device = device
    
    def forward(self, images, batch_size=32, num_splits=10):
        """
        Args:
            images: a tensor of images, whose inception score will be calculated
            batch_size: batch size used while running the inception_v3 model
            num_splits: number of splits used in the inception score calculations
        Returns:
            inception score of the given images
        """
        with torch.no_grad():
            all_preds = torch.zeros((len(images), 1000))
            ind = 0

            images_std_0, images_mean_0 = torch.std_mean(images[:,0,:,:])
            images_std_1, images_mean_1 = torch.std_mean(images[:,1,:,:])
            images_std_2, images_mean_2 = torch.std_mean(images[:,2,:,:])

            images = normalize(images, mean=(images_mean_0, images_mean_1, images_mean_2), std=(images_std_0, images_std_1, images_std_2))

            for idx in range(0, len(images), batch_size):
                images_batch = resize(images[idx:idx+batch_size].to(self.device), [299, 299])

                preds = self.model(images_batch)
                preds = softmax(preds, dim=1)
                all_preds[ind:ind+preds.shape[0]] = preds
                ind = ind + preds.shape[0]
                
            split_results = torch.zeros(num_splits)
            ind = 0
            split_size = ceil(all_preds.shape[0] / num_splits)
            for i in range(num_splits):
                preds = all_preds[ind:ind+split_size]
                ind = ind + split_size
                p_y = torch.mean(preds, dim=0, keepdims=True)
                #split_results[i] = torch.exp(kl_div(preds, p_y, reduction='batchmean'))
                split_results[i] = torch.exp(torch.sum(preds * (torch.log(preds) - torch.log(p_y))) / preds.shape[0])

            return torch.std_mean(split_results)