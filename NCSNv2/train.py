import os
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from copy import deepcopy
from models.models import get_model
from datasets.datasets import get_dataloader
from utils import get_name, parse_arguments, calculate_noise_scales


def train(model, train_loader, optimizer, device, args):
    """
    Trains the model and saves it every 5000 iteration.
    :param model: The model to be trained.
    :param train_loader: The DataLoader object of train dataset.
    :param optimizer: A torch.optim optimizer.
    :param device: Torch device (torch.device('cpu') or torch.device('cuda')).
    :param args: Arguments.
    """
    save_folder = os.path.join('model_saves', get_name(args))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = os.path.join(save_folder, 'checkpoint{}')

    # All noise scales are obtained. scale_idx will be used for the random sampling of the noise scales during training.
    noise_scales = calculate_noise_scales(args.sigma1, 0.01, args.L)
    scale_idx = np.arange(noise_scales.shape[0])

    iteration = 0

    # Exponential Moving Average is taken during checkpoints to calculate FID scores for stability.
    model_ema = get_model(args.model)

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0

        for images, _ in tqdm(train_loader):

            images = images.to(device)

            optimizer.zero_grad()
            np.random.shuffle(scale_idx)

            # The noise scales are randomly sampled.
            scales = torch.from_numpy(noise_scales[scale_idx[:images.size(0)]]).to(device)
            scales = scales.view(-1, 1, 1, 1)

            noise = torch.randn_like(images)

            corrupted_images = images + noise * scales

            # Scores are predicted by the model.
            pred_sc = model(corrupted_images)

            # The target tensor is calculated.
            direction = (images - corrupted_images) / scales

            # We were not sure if they use mean or sum as reduction. Therefore, it is specified with arguments.
            if args.reduction == 'mean':
                loss = F.mse_loss(pred_sc, direction)
            elif args.reduction == 'sum':
                l2_loss = torch.sum((pred_sc - direction) ** 2, (1, 2, 3))
                loss = torch.mean(l2_loss)
            else:
                raise ValueError('Unknown reduction method:', args.reduction)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            iteration += 1
            if iteration % 5000 == 0:
                if iteration == 5000:
                    model_ema = deepcopy(model)
                else:
                    # The Exponential Moving Average is only used for sampling. It does not change the original model.
                    for w1, w2 in zip(model_ema.parameters(), model.parameters()):
                        w1.data.copy_(w1.data * 0.9 + 0.1 * w2.data)

                torch.save({'epoch': epoch,
                            'model': model.state_dict(),
                            'model_ema': model_ema.state_dict(),
                            'optimizer': optimizer.state_dict(), }, save_path.format(iteration))

        train_loss = total_train_loss / len(train_loader)
        print('Epoch: {}, loss: {}'.format(epoch, train_loss))


def main():
    args = parse_arguments()
    device = torch.device('cuda' if args.cuda else 'cpu')

    train_loader = get_dataloader(args.dataset, True, args.batch_size)
    model = get_model(args.model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train(model, train_loader, optimizer, device, args)


if __name__ == '__main__':
    main()
