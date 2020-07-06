import torch.optim as optim
import torchvision.transforms as transforms
import argparse
import time
import datetime
import os
import numpy as np
from marginGAN import *
from utils import *
from dataset import *

# Create arguments to set hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size","--bs","--batch",type=int,default=50)
parser.add_argument("--seed",type=int,default=0)
parser.add_argument("--num_epochs","--epoch",type=int,default=10000)
parser.add_argument("--lrC",type=float,default=0.01)
parser.add_argument("--momentum",type=float,default=0.8)
parser.add_argument("--device",type=str,default="cuda:0")
args = parser.parse_args()

batch_size = args.batch_size
seed = args.seed
num_epochs = args.num_epochs
lrC = args.lrC
momentum = args.momentum
device = args.device

# Select GPU/CPU
if torch.cuda.is_available():
    device = torch.device(device)
else:
    device = torch.device("cpu")

# Provide manual seed for reproducibility
torch.manual_seed(seed)

# Call the datasets
dataset = torchvision.datasets.MNIST("./data",train=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))]),target_transform=None,download=True)
testset = torchvision.datasets.MNIST("./data",train=False,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))]),target_transform=None,download=True)

# Divide the dataset into train and validation
val_size = 10000
train_size = 60000-val_size
trainset, _ = torch.utils.data.random_split(dataset, [train_size, val_size])

# Load the test set
testloader = torch.utils.data.DataLoader(
					dataset=testset,
					batch_size=batch_size,
					shuffle=False)

for label_size in [100,600,1000,3000]:
    # Divide and load the labeled dataset
    labeled_set, _ = divide_dataset(trainset, label_size)
    trainloader = torch.utils.data.DataLoader(dataset=labeled_set,batch_size=batch_size,shuffle=True)
    C = Classifier()
    C = C.to(device)
    optimizer = optim.SGD(C.parameters(),lr=lrC,momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        C.train()
        # Training loop
        for i,(image,label) in enumerate(trainloader,0):
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            outputs, _ = C(image)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i+1) % 10 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
        correct = 0
        total = 0
        C.eval()
        # Evaluation loop
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs, _ = C(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            # Break when the required accuracy levels are reached    
            if label_size == 100 and correct/total > 0.80:
                break
            elif label_size == 600 and correct/total > 0.93:
                break
            elif label_size == 1000 and correct/total > 0.95:
                break
            elif label_size == 3000 and correct/total > 0.97:
                break

        print('[',epoch+1,']','Accuracy: %d %%' % (
                100 * correct / total), correct,"/", total)
        
    save_path = os.path.join("pretrained_classifiers","pre_cls_label_model_"+str(label_size)+".pt")
    torch.save(C, save_path)
    
    
