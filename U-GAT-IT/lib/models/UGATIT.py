import torch.nn.functional as F

from lib.models.generator import Generator
from lib.models.discriminator import LocalDiscriminator, GlobalDiscriminator
from lib.utils import RhoClipper
from lib.loss.losses import *

from lib.base.base_model import BaseModel


class UGATIT(BaseModel):
    def __init__(self, adv_weight, cycle_weight, idt_weight, cam_weight, reflection=True):
        super(UGATIT, self).__init__()

        self.G_AB = Generator(bias=False, reflection=reflection)
        self.G_BA = Generator(bias=False, reflection=reflection)

        self.D_AG = GlobalDiscriminator(reflection=reflection)
        self.D_AL = LocalDiscriminator(reflection=reflection)

        self.D_BG = GlobalDiscriminator(reflection=reflection)
        self.D_BL = LocalDiscriminator(reflection=reflection)

        self.rho_clipper = RhoClipper(0, 1)
        self.mse_loss = LossWithValue(loss_fn=nn.MSELoss())
        self.bce_loss = LossWithValue(loss_fn=nn.BCEWithLogitsLoss())
        self.l1_loss = nn.L1Loss()

        self.adv_weight = adv_weight
        self.cyc_weight = cycle_weight
        self.idt_weight = idt_weight
        self.cam_weight = cam_weight

    def get_generator_parameters(self):
        return list(self.G_AB.parameters()) + list(self.G_BA.parameters())

    def get_discriminator_parameters(self):
        return list(self.D_AL.parameters()) + list(self.D_AG.parameters()) +\
               list(self.D_BL.parameters()) + list(self.D_BG.parameters())

    def forward(self, domain_A, domain_B):
        fake_B, _ = self.G_AB(domain_A)
        fake_A, _ = self.G_BA(domain_B)

        return fake_A, fake_B

    def backward_generators(self, domain_A, domain_B):
        fake_B, logits_cam_AB = self.G_AB(domain_A)
        fake_A, logits_cam_BA = self.G_BA(domain_B)

        D_AG_out_fake, D_AG_logits_fake = self.D_AG(fake_A)
        D_AL_out_fake, D_AL_logits_fake = self.D_AL(fake_A)

        D_BG_out_fake, D_BG_logits_fake = self.D_BG(fake_B)
        D_BL_out_fake, D_BL_logits_fake = self.D_BL(fake_B)

        rec_A, _ = self.G_BA(fake_B)
        rec_B, _ = self.G_AB(fake_A)

        id_A, logits_cam_AA = self.G_BA(domain_A)
        id_B, logits_cam_BB = self.G_AB(domain_B)

        # Adversarial loss
        adv_loss_A = self.mse_loss(D_AG_out_fake, 1.0) + self.mse_loss(D_AL_out_fake, 1.0)
        adv_loss_B = self.mse_loss(D_BG_out_fake, 1.0) + self.mse_loss(D_BL_out_fake, 1.0)
        cam_adv_loss_A = self.mse_loss(D_AG_logits_fake, 1.0) + self.mse_loss(D_AL_logits_fake, 1.0)
        cam_adv_loss_B = self.mse_loss(D_BG_logits_fake, 1.0) + self.mse_loss(D_BL_logits_fake, 1.0)

        # Cycle loss
        cycle_loss_ABA = self.l1_loss(domain_A, rec_A)
        cycle_loss_BAB = self.l1_loss(domain_B, rec_B)

        # Identity loss
        identity_loss_A = self.l1_loss(domain_A, id_A)
        identity_loss_B = self.l1_loss(domain_B, id_B)

        # CAM Loss
        cam_loss_AB = self.bce_loss(logits_cam_AB, 1.0) + self.bce_loss(logits_cam_BB, 0.0)
        cam_loss_BA = self.bce_loss(logits_cam_BA, 1.0) + self.bce_loss(logits_cam_AA, 0.0)

        adv_loss = self.adv_weight * (adv_loss_A + adv_loss_B + cam_adv_loss_A + cam_adv_loss_B)
        cyc_loss = self.cyc_weight * (cycle_loss_ABA + cycle_loss_BAB)
        idt_loss = self.idt_weight * (identity_loss_A + identity_loss_B)
        cam_loss = self.cam_weight * (cam_loss_AB + cam_loss_BA)
        loss = adv_loss + cyc_loss + idt_loss + cam_loss
        return loss, fake_A, fake_B, rec_A, rec_B

    def backward_discriminators(self, domain_A, domain_B):
        fake_B, logits_cam_AB = self.G_AB(domain_A)
        fake_A, logits_cam_BA = self.G_BA(domain_B)

        D_AG_out_fake, D_AG_logits_fake = self.D_AG(fake_A.detach())
        D_AL_out_fake, D_AL_logits_fake = self.D_AL(fake_A.detach())

        D_AG_out_real, D_AG_logits_real = self.D_AG(domain_A)
        D_AL_out_real, D_AL_logits_real = self.D_AL(domain_A)

        D_BG_out_fake, D_BG_logits_fake = self.D_BG(fake_B.detach())
        D_BL_out_fake, D_BL_logits_fake = self.D_BL(fake_B.detach())

        D_BG_out_real, D_BG_logits_real = self.D_BG(domain_B)
        D_BL_out_real, D_BL_logits_real = self.D_BL(domain_B)

        cam_adv_loss_A = self.mse_loss(D_AG_logits_fake, 0.0) + self.mse_loss(D_AL_logits_fake, 0.0) + \
                         self.mse_loss(D_AG_logits_real, 1.0) + self.mse_loss(D_AL_logits_real, 1.0)

        cam_adv_loss_B = self.mse_loss(D_BG_logits_fake, 0.0) + self.mse_loss(D_BL_logits_fake, 0.0) + \
                         self.mse_loss(D_BG_logits_real, 1.0) + self.mse_loss(D_BL_logits_real, 1.0)

        adv_loss_d_A = self.mse_loss(D_AG_out_fake, 0.0) + self.mse_loss(D_AL_out_fake, 0.0) + \
                       self.mse_loss(D_AG_out_real, 1.0) + self.mse_loss(D_AL_out_real, 1.0)

        adv_loss_d_B = self.mse_loss(D_BG_out_fake, 0.0) + self.mse_loss(D_BL_out_fake, 0.0) + \
                       self.mse_loss(D_BG_out_real, 1.0) + self.mse_loss(D_BL_out_real, 1.0)

        loss = cam_adv_loss_A + cam_adv_loss_B + adv_loss_d_A + adv_loss_d_B
        return loss