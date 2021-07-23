import sys
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod

from torch.utils import tensorboard

from lib.utils import code_backup, load_checkpoint
from lib.optim import build_lr_scheduler
import lib.models as models


class BaseTrainer(ABC):
    def __init__(self, config, resume, train_loader, save_dir, log_dir, val_loader=None):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.do_validation = self.config.trainer.val
        self.start_epoch = 1

        # MODEL
        self.model = getattr(models, config.arch.type)(**config.arch.args)

        # LOAD PRETRAINED NETWORK
        checkpoint = self.config.trainer.get('pretrained_network', False)
        if checkpoint:
            checkpoint = load_checkpoint(checkpoint)
            self.model.load_pretrained_weights(checkpoint['state_dict'])
            del checkpoint

        # FREEZE LAYERS
        frozen_layers = self.config.arch.get('frozen_layers', None)
        if frozen_layers is not None:
            self.model.set_trainable_specified_layers(frozen_layers, is_trainable=False)

        # SETTING THE DEVICE
        self.device, self.available_gpus = self._get_available_devices()
        self.model.to(self.device)

        # CONFIGS
        self.epochs = self.config.trainer.epochs
        self.save_period = self.config.trainer.save_period

        # OPTIMIZER
        self.optimizer_g = torch.optim.Adam(self.model.get_generator_parameters(),
                                            **self.config.optimizer.generator.args)
        self.optimizer_d = torch.optim.Adam(self.model.get_discriminator_parameters(),
                                            **self.config.optimizer.discriminator.args)

        self.lr_scheduler_g = build_lr_scheduler(self.optimizer_g,
                                                 lr_scheduler=self.config.lr_scheduler.generator.type,
                                                 max_epoch=self.epochs,
                                                 **self.config.lr_scheduler.generator.args)
        self.lr_scheduler_d = build_lr_scheduler(self.optimizer_d,
                                                 lr_scheduler=self.config.lr_scheduler.discriminator.type,
                                                 max_epoch=self.epochs,
                                                 **self.config.lr_scheduler.discriminator.args)

        # CHECKPOINTS & TENSOBOARD
        self.writer = tensorboard.SummaryWriter(log_dir)
        self.checkpoint_dir = save_dir / 'checkpoints'
        self.visualize_dir = save_dir / 'images'
        self.code_dir = save_dir / 'code'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.visualize_dir.mkdir(parents=True, exist_ok=True)
        self.code_dir.mkdir(parents=True, exist_ok=True)
        code_backup(self.code_dir)

        self.wrt_mode, self.wrt_step = 'train_', 0

        if resume:
            self._resume_checkpoint(resume)

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            torch.cuda.empty_cache()
            # Train epoch
            self.wrt_mode = 'train'
            self._train_epoch(epoch)

            # Print Training Summary
            sys.stdout.description('\n' + self._training_summary(epoch-1)+'\n')

            # DO VALIDATION IF SPECIFIED
            if self.do_validation and epoch % self.config.trainer.val_per_epochs == 0 and self.val_loader is not None:
                sys.stdout.description('\n\n###### EVALUATION ######'+'\n')
                self.wrt_mode = 'val'
                self._valid_epoch(epoch)

                # Print Validation Summary
                sys.stdout.description('\n' + self._validation_summary(epoch-1) + '\n')

            # SAVE CHECKPOINT
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)

    def write_item(self, name, value, step):
        self.writer.add_scalar(f'{self.wrt_mode}/{name}', value, step)

    def _save_checkpoint(self, epoch, is_best=False, remove_module_from_keys=True):
        state = {
            'arch': type(self.model).__name__,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
            'lr_scheduler_g': self.lr_scheduler_g.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict(),
            'lr_scheduler_d': self.lr_scheduler_d.state_dict(),
            'config': self.config
        }
        if remove_module_from_keys:
            # remove 'module.' in state_dict's keys
            state_dict = state['state_dict']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]
                new_state_dict[k] = v
            state['state_dict'] = new_state_dict

        if is_best:
            filename = self.checkpoint_dir / 'best_model.pth'
            torch.save(state, filename)
            sys.stdout.description("\nSaving current best: best_model.pth")
        else:
            filename = self.checkpoint_dir / f'checkpoint-epoch{epoch}.pth'
            torch.save(state, filename)
            sys.stdout.description(f'\nSaving a checkpoint: {filename} ...')

    def _resume_checkpoint(self, resume_path):
        print(f'Loading checkpoint : {resume_path}')
        checkpoint = load_checkpoint(resume_path)

        # Load last run info, the model params, the optimizer and the loggers
        self.start_epoch = checkpoint['epoch'] + 1

        if checkpoint['config']['arch']['type'] != self.config.arch.type:
            print({'Warning! Current model is not the same as the one in the checkpoint'})
        self.model.load_pretrained_weights(checkpoint['state_dict'])

        if checkpoint['config']['optimizer']['generator']['type'] != self.config.optimizer.generator.type:
            print({'Warning! Current optimizer is not the same as the one in the checkpoint'})
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g'])

        if checkpoint['config']['optimizer']['discriminator']['type'] != self.config.optimizer.discriminator.type:
            print({'Warning! Current optimizer is not the same as the one in the checkpoint'})
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d'])

        if checkpoint['config']['lr_scheduler']['generator']['type'] != self.config.lr_scheduler.generator.type:
            print({'Warning! Current lr_scheduler is not the same as the one in the checkpoint'})
        self.lr_scheduler_g.load_state_dict(checkpoint['lr_scheduler_g'])

        if checkpoint['config']['lr_scheduler']['discriminator']['type'] != self.config.lr_scheduler.discriminator.type:
            print({'Warning! Current lr_scheduler is not the same as the one in the checkpoint'})
        self.lr_scheduler_d.load_state_dict(checkpoint['lr_scheduler_d'])

        print(f'Checkpoint <{resume_path}> (epoch {self.start_epoch}) was loaded')

    def _get_available_devices(self):
        n_gpu = self.config.n_gpu
        sys_gpu = torch.cuda.device_count()
        if sys_gpu == 0:
            print('No GPUs detected, using the CPU')
            n_gpu = 0
        elif n_gpu > sys_gpu:
            print(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
            n_gpu = sys_gpu
            torch.cuda.empty_cache()

        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        print(f'Detected GPUs: {sys_gpu} Requested: {n_gpu}')
        available_gpus = list(range(n_gpu))
        return device, available_gpus

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    @abstractmethod
    def _valid_epoch(self, epoch):
        raise NotImplementedError

    @abstractmethod
    def _training_summary(self, epoch):
        raise NotImplementedError

    @abstractmethod
    def _validation_summary(self, epoch):
        raise NotImplementedError

    @abstractmethod
    def _log_train_tensorboard(self, step):
        raise NotImplementedError

    @abstractmethod
    def _log_validation_tensorboard(self, step):
        raise NotImplementedError

