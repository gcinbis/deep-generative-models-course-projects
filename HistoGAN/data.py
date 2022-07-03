# implement torch.data.Dataset class
from torch.utils.data import Dataset
import os
from torchvision.io import read_image
from torchvision.transforms import Resize

class AnimeFacesDataset(Dataset):
    def __init__(self, image_dir, transform=Resize((256,256)), device="cpu"):
        self.image_dir = image_dir
        self.transform = transform
        self.image_tensor_list = []
        i = 0
        for root, dirs, files in os.walk(image_dir):
            for ind, file in enumerate(files):
                path = os.path.join(root, file)
                image = read_image(path).to(device)
                image = self.transform(image/255.0) 
                self.image_tensor_list.append(image)

    def __len__(self):
        return len(self.image_tensor_list)

    def __getitem__(self, index):
        return self.image_tensor_list[index] 


def main():
    image_dir = "images"
    dataset = AnimeFacesDataset(image_dir, device="cuda")
    print(dataset.size())

if __name__ == "__main__":
    main()