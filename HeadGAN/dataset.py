from weakref import ref
from torch.utils.data import Dataset
import torch
import numpy as np
from dataset_generator import get_data_batch
import torchvision
class Face3dDataset(Dataset):
    

    def __init__(self, filenames, frame_per_video, is_gpu=True):
        """
        Args:
            filenames (string): Path to the filenames.
        """
        self.filenames = filenames
        self.is_gpu = is_gpu
        self.frame_per_video = frame_per_video


    def __len__(self):
        return len(self.filenames)


    def __getitem__(self, idx):
        # Get video name
        idx = idx % len(self.filenames)
        video_names = self.filenames[idx]
        # process data
        ref_image , ref_3d, face_3d_list, audio_features, gt, unnormalized = get_data_batch([video_names], self.frame_per_video, is_gpu=self.is_gpu, debug_output=False)
        try:
            if(len(ref_image) == 0 and len(ref_3d) == 0 and len(face_3d_list) == 0 and len(audio_features) == 0):
                raise Exception("Empty data")
            ref_image = torch.Tensor(ref_image)
            ref_3d = torch.Tensor(ref_3d)
            face_3d_list = torch.Tensor(face_3d_list)
            audio_features = torch.Tensor(audio_features)
            gt = torch.Tensor(gt)
        except Exception as e:
            return self.__getitem__(idx+1)

        return ((ref_image, ref_3d ,face_3d_list, audio_features), gt, unnormalized)