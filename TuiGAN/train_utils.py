import torch
from matplotlib import pyplot as plt

from image_utils import apply_gaussian_noise, upsample_image, get_output_image


def get_torch_zeros(shape, device='cuda'):
    """
    Generate zero tensor with given shape
    :param shape: array shape
    :param device: sets training device
    :return: zero tensor with given shape
    """
    return torch.zeros(shape, dtype=torch.float32).to(device)


def generate_input(inps, models, model_idx, device='cuda'):
    """
    Generate images for each scale up to model idx.
    :param inps: Real images.
    :param models: model lists. Include generator A-to-B, B-to-A, discriminator A, B with given order.
    :param model_idx: current scale index.
    :param device: sets training device
    :return: previous scale outputs and current scale A-to-B and B-to-A outputs.
    """

    curr_ab = curr_ba = curr_aba = curr_bab = prev_ab = prev_ba = prev_aba = prev_bab = None

    for idx in range(0, model_idx + 1):
        # Models for current scale.
        gen_ab = models[0][idx]
        gen_ba = models[1][idx]

        # First scale has no previous scale so we used zero array instead of generated input.
        # Zero array is most suitable here because in gradient_penalty we generate interpolated image
        #   where alpha value between [0, 1]
        # To don't effect generated image we use zero array.
        prev_ab = upsample_image(curr_ab) if idx != 0 else get_torch_zeros(inps[0][idx].shape, device=device)
        prev_ba = upsample_image(curr_ba) if idx != 0 else get_torch_zeros(inps[1][idx].shape, device=device)
        prev_aba = upsample_image(curr_aba)if idx != 0 else get_torch_zeros(inps[0][idx].shape, device=device)
        prev_bab = upsample_image(curr_bab)if idx != 0 else get_torch_zeros(inps[1][idx].shape, device=device)

        # Gausian noise added to real image.
        curr_ab = gen_ab(apply_gaussian_noise(inps[0][idx]), prev_ab)

        curr_aba = gen_ba(curr_ab, prev_aba)

        # Gausian noise added to real image.
        curr_ba = gen_ba(apply_gaussian_noise(inps[1][idx]), prev_ba)

        curr_bab = gen_ab(curr_ba, prev_bab)

    return prev_ab, prev_ba, prev_aba, prev_bab, curr_ab, curr_ba


def generate_outputs(inps, models, model_idx, device='cuda', NUM_SCALES=5):
    """
    Visualizer for models. Runs models in evaluation mode.
    :param inps: Real images.
    :param models: model lists. Include generator A-to-B, B-to-A, discriminator A, B with given order.
    :param model_idx: current scale index.
    :param device: sets training device
    :return:
    """
    curr_ab = curr_ba = curr_aba = curr_bab = prev_ab = prev_ba = prev_aba = prev_bab = None

    for idx, (m1, m2, m3, m4) in enumerate(zip(*models)):
        # Last layer should be in train mode due to the batchnorm.
        if idx == NUM_SCALES - 1:
            m1.train()
            m2.train()
            m3.eval()
            m4.eval()
        else:
            m1.eval()
            m2.eval()
            m3.eval()
            m4.eval()

    for idx in range(0, model_idx + 1):
        gen_ab = models[0][idx]
        gen_ba = models[1][idx]

        prev_ab = upsample_image(curr_ab) if idx != 0 else get_torch_zeros(inps[0][idx].shape, device=device)
        prev_ba = upsample_image(curr_ba) if idx != 0 else get_torch_zeros(inps[1][idx].shape, device=device)
        prev_aba = upsample_image(curr_aba)if idx != 0 else get_torch_zeros(inps[0][idx].shape, device=device)
        prev_bab = upsample_image(curr_bab)if idx != 0 else get_torch_zeros(inps[1][idx].shape, device=device)

        curr_ab = gen_ab(apply_gaussian_noise(inps[0][idx]), prev_ab)

        curr_aba = gen_ba(curr_ab, prev_aba)

        curr_ba = gen_ba(apply_gaussian_noise(inps[1][idx]), prev_ba)

        curr_bab = gen_ab(curr_ba, prev_bab)

        # print(f'Scale {idx}, Resolution {curr_ba.size()[2]}x{curr_ba.size()[3]}')
        # plt.subplot(1, 2, 1)
        # plt.imshow(get_output_image(curr_ab))
        # plt.subplot(1, 2, 2)
        # plt.imshow(get_output_image(curr_ba))
        # plt.show()

    return get_output_image(curr_ab), get_output_image(curr_ba)


def generate_identity(inps, models, model_idx, device='cuda'):
    """
    Generates A-to-A and B-to-B images to use in identity loss.
    Real A image is given to generator B-to-A to generate A.
    Real B image is given to generator A-to-B to generate B.
    :param inps: Real images.
    :param models: model lists. Include generator A-to-B, B-to-A, discriminator A, B with given order.
    :param model_idx: current scale index.
    :param device: sets training device
    :return:
    """
    curr = None
    prev = get_torch_zeros(inps[0].shape, device=device)
    for idx in range(model_idx + 1):
        if idx != 0:
            prev = upsample_image(curr)
        gen = models[idx]
        curr = gen(inps[idx], prev)
    return curr
