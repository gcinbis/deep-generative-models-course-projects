import os.path
from pathlib import Path

import torch

from models import ScaleAwareGen, MarkovianDiscriminator


def save_model(model, name, dir_path):
    """
    Save model to given path.
    :param model: model object
    :param name: model file name to save
    :param dir_path: model directory.
    :return:
    """
    model_path = os.path.join(dir_path, f"{name}.pth")
    torch.save(model, model_path)


def load_model(name, dir_path, device='cuda'):
    """
    Load pytorch model from given path.
    :param device: device to load models.
    :param model: model object
    :param name: model file name
    :param dir_path: model directory
    :return: loaded model
    """
    model_path = os.path.join(dir_path, f"{name}.pth")

    if not os.path.exists(model_path):
        return None

    model = torch.load(model_path, map_location=device)

    return model


def save_models(list_g_ab, list_g_ba, list_d_a, list_d_b, data_name):
    """
    Save models to models directory with its model name and scale name.
    :param list_g_ab: A-to-B generator list.
    :param list_g_ba: B-to-A generator list.
    :param list_d_a: A discriminator list.
    :param list_d_b: B discriminator list.
    :param data_name: Name of input image.
    :return:
    """
    for idx, (ab, ba, a, b) in enumerate(zip(list_g_ab, list_g_ba, list_d_a, list_d_b)):
        directory = Path(f'models/{data_name}')
        directory.mkdir(exist_ok=True, parents=True)
        save_model(ab, f'ab{idx}', directory)
        save_model(ba, f'ba{idx}', directory)
        save_model(a, f'a{idx}', directory)
        save_model(b, f'b{idx}', directory)


def load_models(data_name, device='cuda', NUM_SCALES=5):
    """
    Load models from models directory.
    :param device: device to load models
    :param data_name: Name of input image.
    :return:
    """
    res_g_ab, res_g_ba, res_d_a, res_d_b = [], [], [], []
    for idx in range(NUM_SCALES):
        directory = Path(f'models/{data_name}')
        res_g_ab.append(load_model(f'ab{idx}', directory, device=device))
        res_g_ba.append(load_model(f'ba{idx}', directory, device=device))
        res_d_a.append(load_model(f'a{idx}', directory, device=device))
        res_d_b.append(load_model(f'b{idx}', directory, device=device))
    return res_g_ab, res_g_ba, res_d_a, res_d_b


def init_weights(m):
    """
    Initialize weights similar with DCGAN approach. It helps more stable results.
    :param m: model
    :return:
    """
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def create_models(num_scale=4, device='cuda'):
    """
    Creates model arrays for different scales.

    Models are created with default parameters given in paper.
    Filter count is not given in paper. Best results that we achieved is generated with filter count 32.

    :param num_scale: number of scales to generate models.
    :param device: sets training device
    :return: model list
    """
    listg_ab = []
    listg_ba = []
    listd_a = []
    listd_b = []

    for i in range(num_scale):
        ab = ScaleAwareGen(filter_count=32).to(device)
        # Init weights with default values.
        ab.apply(init_weights)
        listg_ab.append(ab)

        ba = ScaleAwareGen(filter_count=32).to(device)
        # Init weights with default values.
        ba.apply(init_weights)
        listg_ba.append(ba)

        a = MarkovianDiscriminator(filter_count=32).to(device)
        # Init weights with default values.
        a.apply(init_weights)
        listd_a.append(a)

        b = MarkovianDiscriminator(filter_count=32).to(device)
        # Init weights with default values.
        b.apply(init_weights)
        listd_b.append(b)
    return listg_ab, listg_ba, listd_a, listd_b


def gradient_penalty(disc, inp, generated, device='cuda'):
    """
    Calculate gradient penalty loss of given images.
    :param disc: discriminator model
    :param inp: real image
    :param generated: fake image
    :param device: sets training device
    :return:
    """
    a = torch.rand(1).to(device)
    interpolated = torch.autograd.Variable(a * inp + (1 - a) * generated, requires_grad=True).to(device)
    out = disc(interpolated)

    grad_out = torch.autograd.Variable(torch.ones(out.size(), dtype=torch.float32), requires_grad=False)
    grad = torch.autograd.grad(outputs=out, inputs=interpolated,
                               grad_outputs=grad_out.to(device),
                               create_graph=True, retain_graph=True, only_inputs=True)[0]
    res = (grad.norm(2, dim=1) - 1) ** 2
    return res.mean()


def total_variation_loss(img):
    """
    This method calculates total variation loss from given image.
    Pixel difference between current and right pixel, and between current and below pixel is used in calculation.
    Sum of squares is calculated and divided by count.
    Division is not specified in paper but without division, total variation loss is very huge and dominates total loss.
    """
    s = img.size()
    count = torch.prod(torch.tensor(s[1:]))

    slide_1 = img[:, :, 1:, :]
    r1 = (img[:, :, :-1, :] - slide_1) ** 2

    slide_2 = img[:, :, :, 1:]
    r2 = (img[:, :, :, :-1] - slide_2) ** 2

    return (r1.sum() + r2.sum()) / count
