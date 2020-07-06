import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pathlib
#from utils import *

import numpy as np
from scipy.stats import entropy

class FakeImageDataset(Dataset):
    def __init__(self, fake_images, transform=None):

        self.fake_images = fake_images
        self.transform = transform

    def __len__(self):
        return len(self.fake_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.fake_images[idx]
        
        if self.transform:
            sample = self.transform(sample)

        return sample

def predict(input, model):

    model.eval()

    #upsample = nn.Upsample(size=(299, 299), mode='bilinear').to(device)
    #output = upsample(input)

    output = model(input)
    output = F.softmax(output, dim=1)

    return output.data.cpu().numpy()

def inception_score_val(generated_images, model, device, batch_size=4):
    N = generated_images.size(0)
    #dummy_labels = torch.tensor(np.zeros((N, 1)))

    _transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = FakeImageDataset(generated_images, _transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    preds = np.zeros((N, 1000))

    for i, data in enumerate(dataloader, 0):
        input = data.to(device)
        batch_size_i = input.size(0)
        preds[i*batch_size:i*batch_size + batch_size_i] = predict(input, model)


    # Now compute the mean kl-div
    split_scores = []

    for k in range(10):
        part = preds[k * (N // 10): (k+1) * (N // 10), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

def imread(filename):
    return np.asarray(Image.open(filename), dtype=np.uint8)[..., :3]

def calculate_score(files, model, batch_size, device):
    model.eval()

    N = len(files)
    if batch_size > N:
        batch_size = N

    pred_arr = np.empty((N, 1000))

    mean = np.array([0.485, 0.456, 0.406])[:, np.newaxis, np.newaxis]
    std = np.array([0.229, 0.224, 0.225])[:, np.newaxis, np.newaxis]

    for i in range(0, N, batch_size):
        if i%1024 == 0 :
            print(i, "/", N)

        start = i
        end = i + batch_size

        images = np.array([imread(str(f)).astype(np.float32) for f in files[start:end]])
        images = images.transpose((0, 3, 1, 2))

        images /= 255
        images -= mean
        images /= std

        batch = torch.from_numpy(images).type(torch.FloatTensor)
        
        batch = batch.to(device)        
        batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)

        pred = predict(batch, model)
        pred_arr[start:end] = pred.reshape(pred.shape[0], -1)
    
    #  Now compute the mean kl-div
    split_scores = []

    for k in range(10):
        part = pred_arr[k * (N // 10): (k+1) * (N // 10), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def calculate_is_score_given_path(path, batch_size, device):
    path = pathlib.Path(path)
    files = list(path.glob('*.jpg')) + list(path.glob('*.png'))

    inception_v3 = torch.load("./data/inception_v3.pt").to(device)

    _mean, _std = calculate_score(files, inception_v3, batch_size, device)

    return _mean, _std



