from PIL import Image
import os
from os import walk
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from fnmatch import fnmatch 


# Our custom dataset, ImageDirectory  inherits Dataset class and overrides the following 3 methods:
# __init__, __getitem__, __len__
class ImageDirectory( Dataset ):
    """ A dataset defined on a directory full of images """

    def __init__(self, directory_path_sharp, directory_path_blur, mode):
        """ 
        Initialize the dataset over a given directory, with  transformations
        to apply to the images.

        """
        super(ImageDirectory, self).__init__()
        
        self.root_dir_sharp = directory_path_sharp
        self.root_dir_blur = directory_path_blur
        self.mode = mode
        if mode == 'train':
            self.transform = transforms.Compose([
                         transforms.Resize(256),
                         transforms.CenterCrop(256),
                         # This makes it into [0,1]
                         transforms.ToTensor(),
                         # This makes it into [-1,1]
                         transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
                         #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
         ])
        elif mode == 'val':
            self.transform = transforms.Compose([
                         transforms.Resize(256),
                         transforms.CenterCrop(256),
                         # This makes it into [0,1]
                         transforms.ToTensor(),
                         # This makes it into [-1,1]
                         transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
                         #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
         ])
        else:
            raise ValueError("Mode [%s] not recognized." % mode)
            
        #(_, _, self.files_sharp) = next(walk(self.root_dir_sharp))
        #(_, _, self.files_blur) = next(walk(self.root_dir_blur))

        extension = "*.png"
        self.files_sharp = list()
        for path, subdirs, files in sorted(os.walk(self.root_dir_sharp)):
            files.sort()
            for name in files:
                if fnmatch(name, extension):
                    self.files_sharp.append(os.path.join(path, name))
                
        self.files_blur = list()
        for path, subdirs, files in sorted(os.walk(self.root_dir_blur)):
            files.sort()
            for name in files:
                if fnmatch(name, extension):
                    self.files_blur.append(os.path.join(path, name)) 
                
    def __getitem__(self, i):
        """ Return the i'th image pair from the dataset."""
        img_name_sharp = self.files_sharp[i]
        img_name_blur = self.files_blur[i]
        
        imgb = Image.open(os.path.join(self.root_dir_blur,img_name_blur)).convert('RGB')
        imgb = self.transform(imgb)
        
        imgs = Image.open(os.path.join(self.root_dir_sharp,img_name_sharp)).convert('RGB')
        imgs = self.transform(imgs)
        
        return imgs, imgb

    def __len__(self):
        """ Return the size of the dataset (number of image pairs) """
        return len(self.files_sharp)


