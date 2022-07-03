import numpy as np
from PIL import Image
from torchvision.utils import save_image
import pathlib
import re
from torchvision import transforms
import torch
import argparse
import os


def group_mask_images_by_ratio(mask_folder_path, output_mask_path, masked_ratio_lower_bound, masked_ratio_upper_bound):
    """
    Group the NVIDIA mask images based on ratio
    :param mask_folder_path: nvidia mask folder path
    :param output_mask_path: output folder that new mask
    :param masked_ratio_lower_bound: lower bound of mask ratio
    :param masked_ratio_upper_bound: upper bound of mask ratio
    :return:
    """
    # Transform mask to 256*256
    mask_transform = transforms.Compose(
        [transforms.Resize(size=(256, 256)), transforms.ToTensor()])

    # Get masks
    path_mask = pathlib.Path(mask_folder_path)
    mask_paths = sorted([file for file in path_mask.glob('*.png')])

    # Check output folder is created
    if not os.path.exists(output_mask_path):
        os.mkdir(output_mask_path)

    # Create Output Dataset Path with ratio
    dataset_path = os.path.join(output_mask_path, 'mask_' + str(masked_ratio_lower_bound) + '_' + str(
        masked_ratio_upper_bound))
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

    # Save mask based on ratio
    for i in range(0, len(mask_paths), 1):
        if i % 100 == 0:
            print('Mask: ' + str(i))

        # Open mask
        mask = Image.open(mask_paths[i])
        mask = mask_transform(mask.convert('RGB'))
        mask = torch.stack([mask, mask], 0)
        numpy_data = np.asarray(mask[0])

        # Get mask ratio
        percent_of_mask = (sum(sum(sum(1 for c in rgb if c != 1.0) for rgb in pixel) for pixel in numpy_data) / (
                256 * 256 * 3)) * 100

        # Save image if mask ratio is between the bounds
        if masked_ratio_upper_bound > percent_of_mask > masked_ratio_lower_bound:
            img_name = re.split("/", str(mask_paths[i]))[-1]
            name = dataset_path + '/' + img_name
            save_image(mask[0], name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_mask_path', default="/home/ibrahim/Desktop/metu/donem2/ceng796/generated_mask",
                        type=str,
                        help='Output folder that new mask')

    parser.add_argument('--mask_folder_path',
                        default="../../irregular_mask/disocclusion_img_mask",
                        type=str, help='Nvidia Mask Folder Path')

    parser.add_argument('--masked_ratio_lower_bound', default=10, type=int,
                        help='lower bound of mask ratio')

    parser.add_argument('--masked_ratio_upper_bound', default=20, type=int,
                        help='upper bound of mask ratio')

    args = parser.parse_args()
    group_mask_images_by_ratio(args.mask_folder_path, args.output_mask_path,
                               args.masked_ratio_lower_bound, args.masked_ratio_upper_bound)
