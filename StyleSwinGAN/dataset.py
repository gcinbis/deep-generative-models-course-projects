from sklearn.utils import shuffle
import torch
import torchvision
from torchvision import transforms


# Sample data from the loader
def get_sample(loader):
    while True:
        for batch in loader:
            yield batch

def get_data_loader(datasetname, root, batch_size, transform):
  if datasetname == 'LSUN':
      dataset = torchvision.datasets.LSUN(
                    root = root,
                    classes = ['church_outdoor_train'],
                    transform=transform
                    )
      dataloader = torch.utils.data.DataLoader(
                              dataset,
                              batch_size = batch_size,
                              num_workers = 2,
                              pin_memory = True,
                              shuffle=False
      )
  
  elif datasetname=='CELEBA':
    dataset = torchvision.datasets.CelebA(
                    root = root,
                    transform = transform,
                    download=True
                    )

    dataloader = torch.utils.data.DataLoader(
                                  dataset,
                                  batch_size = batch_size,
                                  num_workers = 2,
                                  pin_memory = True,
                                  shuffle=False
          )

  elif datasetname == 'CIFAR-10':
      dataset = torchvision.datasets.CIFAR10(
                    root = root,
                    train=True,
                    download=True,
                    transform = transforms.Compose([
                              transforms.Resize(32),
                              transforms.CenterCrop(32),
                              transforms.ToTensor()
                              ])
                    )
      dataloader = torch.utils.data.DataLoader(
                              dataset,
                              batch_size = batch_size,
                              num_workers = 2,
                              pin_memory = True
      )

  else:
    raise ValueError(f'No dataset named {datasetname}!')
  
  return dataloader

if '__name__' == '__main__':
  # Dataset
  datasetname = 'LSUN'
  # Data Root
  root = './'
  # Parameters
  batch_size = 256
  # Get Dataloader
  loader = get_data_loader(datasetname, root, batch_size)
