import torch

from misc.utils import spatial_replication
"""
Implementation of the loss functions defined in the paper at Appendix B.:
https://arxiv.org/pdf/2012.08261.pdf#page=10&zoom=100,66,560
"""
l1 = torch.nn.L1Loss()
bce = torch.nn.BCELoss()

def pixel_losses(real, fake):
    """
    Returns the loss function for pixel-level classification. 
    """
    return l1(real, fake)

def hinge_loss(x,y,h,y_m,discriminator, discriminator_m):
    """
    Returns the loss function for the hinge loss. (L_G^adv)
    """
    D_x_y = discriminator(x,y)[0]
    
    h = spatial_replication(h, y_m)
    D_m = discriminator_m(h,y_m)[0]
    loss = D_x_y + D_m 
    
    return loss

def adversarial_loss_d(x_t,y_t,y_head_t, discriminator):
    """
    Returns the loss function for the adversarial loss. (L_D^adv) and (L_D_m^adv)
    """
    real = discriminator(x_t, y_t)[0]
    fake = discriminator(x_t, y_head_t)[0]

    real_label = torch.ones(real.size(0),1).fill_(1.0).to(real.device)
    fake_label = torch.zeros(fake.size(0),1).fill_(0.0).to(fake.device)

    return bce(fake,real_label) + bce(real,fake_label)

def feature_matching_loss(discriminator, real, fake):
    """
    Returns the loss function for the VGG loss and feature matching loss.
    """
    features = discriminator(real,fake)[1]
    real_feature = features[:,0,:]
    fake_feature = features[:,1,:]

    return l1(real_feature, fake_feature)

def vgg_loss(vgg, real, fake):
    """
    Returns the loss function for the VGG loss and feature matching loss.
    """
    real_feature = vgg(real)
    fake_feature = vgg(fake)

    return l1(real_feature, fake_feature)


def headgan_loss(x_t, h_a, y_t_ref, y_t, y_head_t, y_m_t, y_head_m_t, discriminator, discriminator_m, vgg, lambda_l1 =50, lambda_vgg=10, lambda_fm=10):
    """
    Returns the loss function for the headgan loss.
    """
    L_G_adv = hinge_loss(x_t, y_head_t, h_a, y_head_m_t, discriminator, discriminator_m)
    L_D_adv = adversarial_loss_d(x_t, y_t, y_head_t, discriminator)
    # L_D_m_adv = adversarial_loss_d(x_t, y_m_t, y_head_m_t, discriminator_m)

    L_G_L1 = pixel_losses(y_t, y_head_t)
    L_F_L1 = pixel_losses(y_t, y_t_ref)

    L_G_VGG = vgg_loss(vgg, y_t, y_head_t)
    L_F_VGG = vgg_loss(vgg, y_t, y_t_ref)

    
    L_G_FM = feature_matching_loss(discriminator, y_m_t, y_head_m_t)

    L_G = L_G_adv + lambda_l1 * L_G_L1 + lambda_vgg * L_G_VGG + lambda_fm * L_G_FM + lambda_l1 * L_F_L1 + lambda_vgg * L_F_VGG + lambda_fm * L_F_VGG 
    L_D = L_D_adv
    # L_D_m = L_D_m_adv

    loss = {}
    loss['L_G'] = L_G
    loss['L_D'] = L_D
    # loss['L_D_m'] = L_D_m
    
    return loss