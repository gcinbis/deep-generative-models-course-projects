import torch
from torchvision.models.inception import inception_v3
import torchvision.datasets as dset

# DOWNLOAD THE INCEPTION V3 MODEL FOR IS AND FID CALCULATION
print("Downloading pretrained Inception V3...")
model = inception_v3(pretrained=True)
torch.save(model, "./data/inception_v3.pt")

# DOWNLOAD UNLABELED SET AND TRAIN SET OF THE STL10 DATASET.
print("Downloading STL10...")
unlabeled_set = dset.STL10(root="./data", download=True, split='unlabeled')
train_set = dset.STL10(root="./data", download=True, split='train')

