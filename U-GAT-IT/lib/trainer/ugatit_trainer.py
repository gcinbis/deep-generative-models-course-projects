import time
import numpy as np
from tqdm import tqdm

import torch
from torchvision.utils import save_image

from lib.base import BaseTrainer
from lib.utils import AverageMeter


class UGATITTrainer(BaseTrainer):
    def __init__(self, config, resume, train_loader, save_dir, log_dir, val_loader=None):
        super().__init__(config, resume, train_loader, save_dir, log_dir, val_loader)

        self.train_vis = self.config.trainer.get('visualize_train_batch', False)
        self.val_vis = self.config.trainer.get('visualize_val_batch', False)
        self.vis_count = self.config.trainer.get('vis_count', len(self.train_loader))
        self.log_per_batch = self.config.trainer.get('log_per_batch', int(np.sqrt(self.train_loader.batch_size)))

    def _train_epoch(self, epoch):
        vis_save_dir = self.visualize_dir / 'train' / str(epoch)
        vis_save_dir.mkdir(parents=True, exist_ok=True)
        self.model.train()
        train_vis_count = 0
        tic = time.time()
        self._reset_metrics()

        tbar = tqdm(self.train_loader)
        for batch_idx, data in enumerate(tbar):
            self.data_time.update(time.time() - tic)
            batches_done = (epoch-1) * len(self.train_loader) + batch_idx

            domain_A = data['A'].to(self.device)
            domain_B = data['B'].to(self.device)

            # Update D
            self.optimizer_d.zero_grad()
            loss_d = self.model.backward_discriminators(domain_A, domain_B)
            loss_d.backward()
            self.optimizer_d.step()

            # Update G
            self.optimizer_g.zero_grad()
            loss_g, fake_A, fake_B, rec_A, rec_B = self.model.backward_generators(domain_A, domain_B)
            loss_g.backward()
            self.optimizer_g.step()

            # clip parameter of AdaILN and ILN, applied after optimizer step
            self.model.G_AB.apply(self.model.rho_clipper)
            self.model.G_BA.apply(self.model.rho_clipper)

            # update metrics
            self.loss_g_meter.update(loss_g.item())
            self.loss_d_meter.update(loss_d.item())
            self.batch_time.update(time.time() - tic)
            tic = time.time()

            # Visualize batch & Tensorboard log
            if batch_idx % self.log_per_batch == 0:
                self._log_train_tensorboard(batches_done)
                if train_vis_count < self.vis_count and self.train_vis:
                    train_vis_count += domain_A.shape[0]
                    self._visualize_batch(domain_A, domain_B, fake_A, fake_B, rec_A, rec_B, vis_save_dir, batch_idx)

            tbar.set_description(self._training_summary(epoch))

        self.lr_scheduler_g.step()
        self.lr_scheduler_d.step()

    def _valid_epoch(self, epoch):
        vis_save_dir = self.visualize_dir / 'test' / str(epoch)
        vis_save_dir.mkdir(parents=True, exist_ok=True)
        self.model.eval()
        self._reset_metrics()
        tbar = tqdm(self.val_loader)
        with torch.no_grad():
            val_vis_count = 0
            for batch_idx, data in enumerate(tbar):
                domain_A = data['A'].to(self.device)
                domain_B = data['B'].to(self.device)

                fake_A, fake_B = self.model.forward(domain_A, domain_B)
                rec_A, rec_B = self.model.forward(fake_A, fake_B)

                # TODO: update evaluation metrics
                # self._update_metrics(pred, label)

                # Visualize batch
                if val_vis_count < self.vis_count and self.val_vis:
                    val_vis_count += domain_A.shape[0]
                    self._visualize_batch(domain_A, domain_B, fake_A, fake_B, rec_A, rec_B, vis_save_dir, batch_idx)

                # PRINT INFO
                if batch_idx == len(tbar)-1:
                    tbar.set_description(self._validation_summary(epoch))

            self._log_validation_tensorboard(epoch)
        # TODO: return the validation score
        return

    # TODO Calculate and update metrics
    def _update_metrics(self, pred, gt):
        pass

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.loss_g_meter = AverageMeter()
        self.loss_d_meter = AverageMeter()

    def _log_train_tensorboard(self, step):
        self.write_item(name='loss_G', value=self.loss_g_meter.avg, step=step)
        self.write_item(name='loss_D', value=self.loss_d_meter.avg, step=step)

        for i, opt_group in enumerate(self.optimizer_g.param_groups):
            self.write_item(name=f'Learning_rate_generator_{i}', value=opt_group['lr'], step=self.wrt_step)

        for i, opt_group in enumerate(self.optimizer_d.param_groups):
            self.write_item(name=f'Learning_rate_discriminator_{i}', value=opt_group['lr'], step=self.wrt_step)

    # TODO Tensorboard logs for the validation
    def _log_validation_tensorboard(self, step):
        pass

    def _training_summary(self, epoch):
        return f'TRAIN [{epoch}] ' \
               f'G: {self.loss_g_meter.val:.3f}({self.loss_g_meter.avg:.3f}) | ' \
               f'D: {self.loss_d_meter.val:.3f}({self.loss_d_meter.avg:.3f}) | ' \
               f'LR_G {self.optimizer_g.param_groups[0]["lr"]:.5f} | ' \
               f'LR_D {self.optimizer_d.param_groups[0]["lr"]:.5f} | ' \
               f'B {self.batch_time.avg:.2f} D {self.data_time.avg:.2f}'

    def _validation_summary(self, epoch):
        return f'EVAL [{epoch}] | '\
               f'G: {self.loss_g_meter.val:.3f}({self.loss_g_meter.avg:.3f}) | ' \
               f'D: {self.loss_d_meter.val:.3f}({self.loss_d_meter.avg:.3f})'

    def _visualize_batch(self, real_A, real_B, fake_A, fake_B, rec_A, rec_B, vis_save_dir, step):
        real_A = self.train_loader.dataset.denormalize(real_A, device=self.device)
        real_B = self.train_loader.dataset.denormalize(real_B, device=self.device)
        fake_A = self.train_loader.dataset.denormalize(fake_A, device=self.device)
        fake_B = self.train_loader.dataset.denormalize(fake_B, device=self.device)
        rec_A = self.train_loader.dataset.denormalize(rec_A, device=self.device)
        rec_B = self.train_loader.dataset.denormalize(rec_B, device=self.device)
        vis_img = torch.cat((real_A.cpu().detach(),
                             real_B.cpu().detach(),
                             fake_A.cpu().detach(),
                             fake_B.cpu().detach(),
                             rec_A.cpu().detach(),
                             rec_B.cpu().detach()), dim=-1)
        save_image(vis_img, str(vis_save_dir / f'index_{step}.png'), nrow=1)