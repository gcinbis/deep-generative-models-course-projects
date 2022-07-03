import numpy as np
from torch.utils.data import Dataset


class ToyDataset(Dataset):
    """The Toy Dataset class for 2D points."""
    def __init__(self, inp_path, out_path):
        self.x = np.load(inp_path).astype(np.float32)
        self.y = np.load(out_path)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]
