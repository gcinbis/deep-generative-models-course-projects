import argparse
import torch
from torchvision import transforms
import os
from .pconv_net import *
import re
import pathlib
import random
from torchvision.utils import save_image
from PIL import Image
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# Mean and variance of normalization/unnormalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def unnormalize(x):
    """
    Unnormalize the given image
    :param x: image
    :return: unnormalized image
    """
    x = x.transpose(1, 3)
    x = x * torch.Tensor(STD) + torch.Tensor(MEAN)
    x = x.transpose(1, 3)
    return x


def load_model(model_path, device):
    """
    Load Partial Convolution Based Encoder/Decoder Model
    :param model_path: path of pretrained model
    :param device: cuda/cpu
    :return: pretrained model
    """
    model = PConvUNet().to(device)
    ckpt_dict = torch.load(model_path)
    model.load_state_dict(ckpt_dict['model'], strict=False)
    return model


def create_pc_images(model, gt_folder_path, dest_folder_path, mask_folder_path, is_save_images,
                     is_random_mask_match_used, is_show_images):
    """
    Create Partial Convolution Generated Images for PD-GAN training
    :param model: PC based Model Path
    :param gt_folder_path: ground truth images folder
    :param dest_folder_path: generated images saving folder
    :param mask_folder_path: mask folder path
    :param is_save_images: Boolean Value that specifies save image or not
    :param is_random_mask_match_used : Boolean Value that specifies matches mask and ground truth dataset randomly
    :param is_show_images : Boolean Value that specifies shows images in grid format

    :return:
    """

    # Specifies torch device
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

    # Image/Mask Transformation, Resize them 256*256 for PC
    img_transform = transforms.Compose(
        [transforms.Resize(size=(256, 256)), transforms.ToTensor(),
         transforms.Normalize(mean=MEAN, std=STD)])
    mask_transform = transforms.Compose(
        [transforms.Resize(size=(256, 256)), transforms.ToTensor()])

    folders_in_dataset = []

    # Get folders inside the ground truth folder
    if os.path.isdir(gt_folder_path):
        folders_in_dataset = [f for f in os.listdir(gt_folder_path) if
                              os.path.isdir(os.path.join(gt_folder_path, f))]
    # If no folders inside the ground truth folder, main path consists images
    if len(folders_in_dataset) == 0:
        folders_in_dataset.append('')

    #Load Model
    model = load_model(model, device)
    model.eval()

    for folder in folders_in_dataset:
        dataset_path = os.path.join(gt_folder_path, folder)
        dest_folder = dest_folder_path + '/' + folder
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        path_mask = pathlib.Path(mask_folder_path)
        path_gt = pathlib.Path(dataset_path)

        # get ground truth and mask images
        gt_paths = sorted([file for file in path_gt.glob('*.png')])
        mask_paths = sorted([file for file in path_mask.glob('*.png')])

        for i, data in enumerate(gt_paths):
            #if i % 100 == 0:
            #    print("Image: " + str(i))

            # get image name
            img_name = re.split("/", str(data))[-1]
            img_name = re.split("\.", img_name)[0]

            # Open image and convert it to RGB format
            gt_img = Image.open(data)
            gt_img = img_transform(gt_img.convert('RGB'))

            # choose mask randomly or not
            if is_random_mask_match_used is False:
                mask = Image.open(mask_paths[i])
            else:
                mask = Image.open(mask_paths[random.randint(0, len(mask_paths) - 1)])

            # open mask
            mask = mask_transform(mask.convert('RGB'))

            # prep input for model
            masked_image = gt_img * mask
            masked_image = torch.stack([masked_image, masked_image], 0)
            mask = torch.stack([mask, mask], 0)
            gt_img = torch.stack([gt_img, gt_img], 0)

            # Forward input to the model
            with torch.no_grad():
                output, _ = model(masked_image.to(device), mask.to(device))

            # Prep output for saving process
            output = output.to(torch.device('cpu'))

            # Set image names
            gen_name = dest_folder + "/" + img_name + "_generated.png"
            gt_name = dest_folder + "/" + img_name + "_gt.png"
            mask_name = dest_folder + "/" + img_name + "_mask.png"

            # Save ground truth, masked and generated images
            if is_save_images:
                save_image(unnormalize(output)[0], gen_name)
                save_image(unnormalize(gt_img)[0], gt_name)
                save_image(mask[0], mask_name)

            # Show images in grid format
            if is_show_images:
                grid = make_grid(
                    [unnormalize(masked_image)[0], unnormalize(gt_img)[0], unnormalize(output)[0]
                     ])
                plt.imshow(grid.permute(1, 2, 0))
                plt.title(["Masked Images", "Ground Truth", "Partial Convolution(PC)"])
                plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='../../models/pc_model.pth',
                        help='PC based Model Path')

    parser.add_argument('--gt_folder_path', default="../../goal_dataset/goal_gt", type=str,
                        help='ground truth images folder')

    parser.add_argument('--mask_folder_path',
                        default="../../goal_dataset/goal_mask",
                        type=str, help='Mask folder path')

    parser.add_argument('--dest_folder_path',
                        default="../../goal_dataset/goal_generated",
                        type=str, help='Generated images saving folder')

    parser.add_argument('--is_random_mask_match_used',
                        default=True,
                        type=bool, help='Boolean Value that specifies matches mask and ground truth dataset randomly')

    parser.add_argument('--is_save_images',
                        default=False,
                        type=bool, help='Boolean Value that specifies save image or not')

    parser.add_argument('--is_show_images',
                        default=False,
                        type=bool, help='Boolean Value that specifies show images or not')

    args = parser.parse_args()
    create_pc_images(args.model, args.gt_folder_path, args.dest_folder_path, args.mask_folder_path,
                     args.is_save_images, args.is_random_mask_match_used, args.is_show_images)
