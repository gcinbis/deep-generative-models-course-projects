import torchvision.datasets.celeba as celeba

class CelebA(celeba.CelebA):
    """ A simple wrapper around torch.datasets.celebA.
    We shouldn't need to do anything about the data, but it may be useful to
    store data-dependent things here in the future.
    
     """ 
    pass
