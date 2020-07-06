import os

import numpy as np
import scipy
import torch
import torch.nn.functional as F

from PIL import Image


def seed_rngs(seed):
    """ Seed all random number generators together """
    np.random.seed(seed)
    torch.manual_seed(seed)


def sum_param_norms(model):
    """ Returns the sum of the norm of each layer in a model, useful to ensure no change """
    norm = torch.tensor([torch.norm(x) for x in model.parameters()]).sum().item()
    return norm


def exact_interpolate(x, scaling_factor, exact_size=None, mode='bicubic'):
    """
    A function for performing interpolation with exact (floating point) sizes.

    Args:
        x: a 4D (N, C, H, W) torch tensor representing the image to interpolate
        scaling_factor: a float, e.g. 1.20 for upsampling
        exact_size: a (float, float) tuple, representing the exact computed size of the image,
            e.g. could be (32.38, 61.77) for an input image of size (33, 62)
            if None, the shape of x is used as the exact_size
        mode: a string, upsampling mode compatible with torch.nn.functional.interpolate's mode argument

    Returns:
        interp: the interpolated version of x
        interp_exact_size: the exact size of interp
    """
    if exact_size is None:
        exact_size = tuple(float(d) for d in x.shape[2:4])  # (H, W)
    interp_exact_size = tuple(scaling_factor * d for d in exact_size)
    interp_rounded_size = tuple(round(d) for d in interp_exact_size)
    # suppress the warning about align_corners by specifying it
    if mode in ['linear', 'bilinear', 'bicubic', 'trilinear']:
        interp = F.interpolate(x, size=interp_rounded_size, mode=mode, align_corners=False)
    else:
        interp = F.interpolate(x, size=interp_rounded_size, mode=mode)
    return interp, interp_exact_size


def resize_long_edge(images, target_size, mode='bicubic'):
    """
    Given an image(s), resize its long edge to correspond to the
    target_size while preserving aspect ratio. e.g. a 400x300
    image with target_size=100 will be resized to 100x75.

    Args:
        images: a 4D (N, C, H, W) torch tensor, representing the images
        target_size: an integer, target long edge size
        mode: a string, upsampling mode compatible with torch.nn.functional.interpolate's mode argument

    Returns:
        resized: the resized 4D tensor
    """
    h, w = tuple(images.shape[2:4])
    scaling_factor = target_size / h if h > w else target_size / w
    resized, _ = exact_interpolate(images, scaling_factor, mode=mode)
    return resized


def create_scale_pyramid(img, scaling_factor, num_scales, mode='bicubic'):
    exact_size = tuple(float(d) for d in img.shape[2:4])  # (N, C, H, W) -> (H, W)
    scaled_images, exact_scale_sizes = [img], [exact_size]
    for i in range(num_scales - 1):
        img, exact_size = exact_interpolate(img, scaling_factor, exact_size, mode)
        scaled_images.append(img)
        exact_scale_sizes.append(exact_size)
    return scaled_images, exact_scale_sizes


def np_image_to_normed_tensor(img_uint):
    """
    Convert a [0,255] uint8 (H, W, C) numpy image input to
    an [-1, 1] float32 (1, C, H, W) torch tensor (default device)
    """
    # convert [0, 255] uint8 (H, W, C) to [-1, 1] float32 (1, C, H, W)
    rescaled = (img_uint.astype('float32') / 127.5) - 1.0
    chw = np.transpose(rescaled, (2, 0, 1))
    return torch.from_numpy(np.expand_dims(chw, axis=0))


def normed_tensor_to_np_image(img_float):
    """ Inverse of np_image_to_normed_tensor """
    chw = np.squeeze(img_float.detach().cpu().numpy())
    hwc = np.transpose(chw, (1, 2, 0))
    clipped = np.clip(hwc, -1.0, 1.0)
    return ((clipped + 1.0) * 127.5).astype('uint8')


def gradient_penalty(discriminator, fake_batch, real_batch):
    """
    Calculate the gradient penalty given a discriminator and fake + real batches.
    Should work in all sorts of settings, FC, CNN, PatchGAN etc.
    """

    # how to calculate this loss is not very clear in this context...
    # In the case of a scalar discr. output, what should be done is simply
    # norm the gradient (image-shaped) across the channel axis, and take
    # the mean across all pixels.
    # In this case, the output of the critic (discr) is an image (PatchGAN).
    # If we take its mean to obtain a scalar and then apply the same approach
    # as the scalar output discr., it seems to suppress the penalty twice
    # (as if the mean was applied twice). Instead, taking the sum of the
    # output allows us to apply the mean only once, which we believe is the
    # proper normalization.

    batch_size = real_batch.shape[0]
    # take samples from the line between the real and generated data points
    # for use in the gradient penalty (Impr. Training of WGANs)
    epsilons = torch.rand(batch_size, device=real_batch.device)
    # noinspection PyTypeChecker
    grad_sample = epsilons * real_batch + (1.0 - epsilons) * fake_batch
    # use the samples to calculate gradient norm
    f_grad_sample = discriminator(grad_sample).sum()
    grad, = torch.autograd.grad(f_grad_sample, grad_sample, create_graph=True, retain_graph=True)
    grad_loss = ((torch.norm(grad, 2, dim=1) - 1) ** 2).mean()  # mean over batch
    return grad_loss


def optimization_step(loss, optimizer, scheduler, loss_records):
    """ Perform common steps related to optimization (usually at the end of the training loop) """
    loss.backward()  # back propagate
    optimizer.step()  # gradient descent
    scheduler.step()  # lr scheduler step
    loss_records.append(loss.item())  # record loss


def has_same_architecture(a, b):
    """
    Checks for architectural equality between models using parameter names and shapes,
    essentially testing for state_dict compatibility between the two inputs
    """
    for (a_name, a_var), (b_name, b_var) in zip(a.named_parameters(), b.named_parameters()):
        if a_name != b_name or a_var.shape != b_var.shape:
            return False
    return True


def initialize_net(net, prev_nets):
    """
    Used for initializing a network from the previous scale and
    adding it to the list of similar networks.

    Args:
        net: a neural network (a generator or critic at a single scale)
        prev_nets: list of the networks on the previous scales, not containing net yet
    """
    # if possible, initialize with weights from the lower layer
    if prev_nets and has_same_architecture(net, prev_nets[-1]):
        net.load_state_dict(prev_nets[-1].state_dict())
    # set train mode & add to list
    net.train()
    prev_nets.append(net)


def load_image(image_path, max_input_size=None, device='cpu', verbose=False):
    """
    Load an image from the given path as a (1, C, H, W) normalized torch tensor.
    If any of its edges is longer than max_input_size, the image will be resized.
    If verbose=True, the function will print information about image shapes.
    """
    # both np uint and tensor versions
    orig_img_uint = np.array(Image.open(image_path).convert('RGB'))
    orig_img = np_image_to_normed_tensor(orig_img_uint)
    input_img = orig_img

    if max_input_size is not None:
        # resize the image to max size if necessary
        orig_h, orig_w, _ = orig_img_uint.shape
        if orig_h > max_input_size or orig_w > max_input_size:
            input_img = torch.clamp(resize_long_edge(orig_img, max_input_size), -1, 1)

    # print info about sizes if verbose
    if verbose:
        oh, ow = tuple(orig_img.shape[2:4])
        if orig_img.shape != input_img.shape:
            ih, iw = tuple(input_img.shape[2:4])
            print('Resized image from {}x{} to {}x{}'.format(oh, ow, ih, iw))
        else:
            print('Image size is {}x{}'.format(oh, ow))

    return input_img.to(device)


def load_with_reverse_pyramid(image_path, max_input_size, scaling_factor, num_scales,
                              mode='bicubic', device='cpu', verbose=False):
    # load the image and create the scale pyramid
    input_img = load_image(image_path, max_input_size, device, verbose)
    scaled_inputs, scaled_exact_sizes = create_scale_pyramid(input_img, scaling_factor, num_scales, mode)
    # reverse both since we start from the coarsest scale
    scaled_inputs.reverse()
    scaled_exact_sizes.reverse()
    return scaled_inputs, scaled_exact_sizes

