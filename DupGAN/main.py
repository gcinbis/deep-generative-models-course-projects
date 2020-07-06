import argparse
import logging
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from train import EncoderClassifierTrainer, GeneratorDiscriminatorTrainer
from utils import mnist_dataset, svhn_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default='cuda')
    parser.add_argument('--log_level', default='INFO')
    parser.add_argument('--encoder_classifier_tensorboard_root', default=os.path.join('runs', ''))
    parser.add_argument('--generator_discriminator_tensorboard_root', default=os.path.join('gdruns', ''))

    parser.add_argument('--datasets_root', default=os.path.join('datasets', ''))
    parser.add_argument('--load_all_svhn_to_device', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--load_all_mnist_to_device', type=lambda x: (str(x).lower() == 'true'), default=False)

    parser.add_argument('--ckpt_root', default=os.path.join('ckpts', ''))
    parser.add_argument('--encoder_classifier_adam_lr', type=float, default=0.0002)
    parser.add_argument('--generator_adam_lr', type=float, default=0.0002)
    parser.add_argument('--discriminator_adam_lr', type=float, default=0.0002)
    parser.add_argument('--encoder_classifier_adam_beta1', type=float, default=0.5)
    parser.add_argument('--encoder_classifier_adam_beta2', type=float, default=0.999)
    parser.add_argument('--generator_adam_beta1', type=float, default=0.5)
    parser.add_argument('--generator_adam_beta2', type=float, default=0.999)
    parser.add_argument('--discriminator_adam_beta1', type=float, default=0.5)
    parser.add_argument('--discriminator_adam_beta2', type=float, default=0.999)
    parser.add_argument('--dupgan_alpha', type=float, default=10.0) # a loss multiplier from dupgan paper
    parser.add_argument('--dupgan_beta', type=float, default=0.2)
    parser.add_argument('--encoder_classifier_confidence_threshold', type=float, default=0.99)
    parser.add_argument('--encoder_classifier_num_epochs', type=int, default=30)
    parser.add_argument('--generator_discriminator_num_epochs', type=int, default=100)
    parser.add_argument('--encoder_classifier_ckpt_file', default='ckpts/encoder_classifier_16.tar')
    parser.add_argument('--encoder_classifier_experiment_name', default='encoder_classifier')
    parser.add_argument('--skip_encoder_classifier_train', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--skip_generator_discriminator_train', type=lambda x: (str(x).lower() == 'true'),
                        default=False)
    parser.add_argument('--generator_discriminator_ckpt_file', default=None)
    parser.add_argument('--batch_size', type=int, default=64)

    params = parser.parse_args()

    logging.getLogger().setLevel(params.log_level)

    # tensorboard monitoring
    encoder_classifier_summary_writer = SummaryWriter(log_dir=params.encoder_classifier_tensorboard_root)
    generator_discriminator_summary_writer = SummaryWriter(log_dir=params.generator_discriminator_tensorboard_root)

    # dataloading
    svhn_trainsplit = svhn_dataset(params.datasets_root, "train")
    svhn_testsplit = svhn_dataset(params.datasets_root, "test")
    mnist_testsplit = mnist_dataset(params.datasets_root, False)
    mnist_trainsplit = mnist_dataset(params.datasets_root, True)


    logging.info('Loading %s as ckpt_file', params.encoder_classifier_ckpt_file)
    encoder_classifier_trainer = EncoderClassifierTrainer(params.device, params,
                                       svhn_trainsplit, svhn_testsplit, mnist_trainsplit, mnist_testsplit,
                                       encoder_classifier_summary_writer, ckpt_file=params.encoder_classifier_ckpt_file,
                                       ckpt_root=params.ckpt_root)

    if not params.skip_encoder_classifier_train:
        logging.info('begin encoder_classifier pretraining...')
        encoder_classifier_trainer.train_until_epoch(params.encoder_classifier_num_epochs)

    encoder_classifier = encoder_classifier_trainer.encoder_classifier
    encoder_classifier_optimizer = encoder_classifier_trainer.optimizer

    logging.info('Loading %s as ckpt_file', params.generator_discriminator_ckpt_file)
    generator_discriminator_trainer = GeneratorDiscriminatorTrainer(params.device, params, encoder_classifier,
                                                                    encoder_classifier_optimizer, svhn_trainsplit,
                                                                    svhn_testsplit, mnist_trainsplit, mnist_testsplit,
                                                                    generator_discriminator_summary_writer,
                                                                    ckpt_file=params.generator_discriminator_ckpt_file,
                                                                    ckpt_root=params.ckpt_root)

    if not params.skip_generator_discriminator_train:
        logging.info('begin core dupgan training...')
        generator_discriminator_trainer.train_until_epoch(params.generator_discriminator_num_epochs)


    logging.info('training complete!')



