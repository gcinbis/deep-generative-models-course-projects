import math
import os.path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def read_domains(dir_path, image_A_file='A.jpg', image_B_file='B.jpg',
                 resize=True, resize_shape=(250, 250), method=Image.BICUBIC):
    """Read two image domains from the given dir_path"""
    image_A_path = os.path.join(dir_path, image_A_file)
    image_B_path = os.path.join(dir_path, image_B_file)

    image_A = Image.open(image_A_path).convert('RGB')
    image_B = Image.open(image_B_path).convert('RGB')

    if resize and image_A.size != resize_shape:
        image_A = image_A.resize(resize_shape, resample=method)

    if resize and image_B.size != resize_shape:
        image_B = image_B.resize(resize_shape, resample=method)

    return np.array(image_A), np.array(image_B)


def normalize_image_to_tensor(img):
    """
    Normalizes the given np image array and converts it to a tensor.
    Basically, it maps [0, 255] RGB values to [-1, 1]

    :param img: np image array with shape (H, W, C)
    :return: normed image tensor with shape (1, C, H, W)
    """

    normed_image = torch.Tensor((img / 127.5) - 1)
    normed_image = normed_image.clamp(-1, 1)

    normed_image = normed_image.permute(2, 0, 1)

    return normed_image.unsqueeze(0)


def denormalize_image_from_tensor(normed_img):
    """
    Denormalizes the given image tensor, as opposite to normalize_image_to_tensor

    :param img: normed image tensor with shape (1, C, H, W)
    :return:
    """

    img = normed_img.clamp(-1, 1).squeeze(0).permute(1, 2, 0)
    img = (img + 1) * 127.5

    return img.numpy().astype('uint8')


def construct_scale_pyramid(image, scale_factor=0.75, N=4, method='bicubic'):
    """
    Constructs the scale pyramid

    :param image: normed image tensor with shape (N, C, H, W)
    :param scale_factor: float that indicates scale factor
    :param N: int that indicates the number of needed scaled image
    :param method: interpolation method while downscaling the image
    :return: Scale pyramid, list of normed image tensors
    """
    pyramid = [image]

    for i in range(N):
        scale = scale_factor ** (i+1)
        scale_shape = (image.shape[2] * scale, image.shape[3]*scale)
        scale_shape = tuple(map(math.ceil, scale_shape))
        downsampled_img = torch.nn.functional.interpolate(image, size=scale_shape, mode=method, align_corners=False)

        pyramid.append(downsampled_img)

    return pyramid


def upsample_image(image, scale_factor=0.75, method='bicubic'):
    """
    Upsample image with the 1/scale_factor by using given method

    :param image: normed image tensor
    :param scale_factor: scale factor
    :param method: interpolation method
    :return: upsampled normed image tensor
    """
    upsample_scale_factor = 1/scale_factor
    scale_shape = (image.shape[2] * upsample_scale_factor, image.shape[3] * upsample_scale_factor)
    scale_shape = tuple(map(math.floor, scale_shape))
    upsampled_img = torch.nn.functional.interpolate(image, size=scale_shape, mode=method, align_corners=False)

    return upsampled_img


def get_output_image(img):
    """
    :param img: image tensor which is on cuda
    :return: PIL image in cpu
    """
    out = denormalize_image_from_tensor(img.detach().cpu())
    out = Image.fromarray(out)
    return out


def apply_gaussian_noise(img, amplifier=0.1):
    """
    Applies gaussian noise to given image.

    :param img: input image to add noise.
    :param amplifier: noise amplifier to control noise.
    :return: noisy image
    """
    noise = torch.randn_like(img)
    return (amplifier * noise) + img


if __name__ == '__main__':
    dir_path = 'data'
    imgA, imgB = read_domains(dir_path, resize=True)
    normed_imgA = normalize_image_to_tensor(imgB)
    pyramid = construct_scale_pyramid(normed_imgA)

    plt.subplot(1,2,1)
    plt.imshow(denormalize_image_from_tensor(pyramid[-2]))
    plt.subplot(1,2,2)
    plt.imshow(denormalize_image_from_tensor(upsample_image(pyramid[-1])))
    plt.show()
