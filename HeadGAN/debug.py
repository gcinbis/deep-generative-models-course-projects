# %%
from dataset import Face3dDataset
from torch.utils.data import DataLoader
import torch
import glob 

from training import training_loop
import os


# %%
train_filelist = list(glob.iglob("data/train/mp4/" + '**/*.mp4', recursive=True))
test_filelist = list(glob.iglob("data/test/mp4/" + '**/*.mp4', recursive=True))


# %%
train_dataset = Face3dDataset(train_filelist, 16, is_gpu=False)
test_filelist = Face3dDataset(test_filelist, 2, is_gpu=False)




# %%
train_dataset = train_dataset
test_dataset = test_filelist
batch_size = 1
epochs = 10
step_size = 200
lr = 0.001
beta1 = 0.5
beta2 = 0.999
use_cuda = True if torch.cuda.is_available() else False
log_dir = "logs"
model_dir = "models"
model_name = "model"
num_workers = 0
checkpoint = "models/model"
checkpoint_epoch = 1
checkpoint_step = 300

training_loop(train_dataset, test_dataset, batch_size, epochs, step_size, lr, beta1, beta2, use_cuda, log_dir, model_dir, model_name, num_workers, checkpoint, checkpoint_step, checkpoint_epoch )


# %%



