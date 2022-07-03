# implement loss functions
import utils
import torch
import numpy as np

# See Eq. 5
# g: generated image, d_score: scalar output of discriminator
def non_sat_generator_loss(g, d_score, hist_t):
    c_loss = hellinger_dist_loss(g, hist_t)
    alpha = 2.0  # See Sec. 5.2 Training details
    # print(alpha*c_loss)
    g_loss = torch.mean(d_score) + alpha*c_loss
    # g_loss = torch.mean(torch.nn.functional.softplus(-d_score)) + alpha*c_loss
    return g_loss 
# R1 regularization applied to discriminator
# Works similar to gradient penalty in wgan-gp
def r1_reg(real_data, real_scores, r1_factor):
    gradients = torch.autograd.grad(outputs=real_scores.mean(), inputs=real_data, create_graph=True, retain_graph=True)[0]
    r1_reg = torch.mean(torch.sum(torch.square(gradients).view(real_data.size(0), -1), dim=1))
    return r1_factor*r1_reg  

# path length regularization introduced in StyleGAN 2 paper
def pl_reg(fake_data, w, target_scale, plr_factor, ema_decay_coeff, device):
    y = torch.randn_like(fake_data) / np.sqrt(fake_data.size(2) * fake_data.size(3))
    y = y.to(device)
    gradients = torch.autograd.grad(outputs=(fake_data*y).sum(), inputs=w, create_graph=True)[0]
    j_norm  = torch.sqrt(torch.sum(torch.square(gradients), dim=1))
    j_norm_mean = torch.mean(j_norm)
    target_scale = (1-ema_decay_coeff)* target_scale + ema_decay_coeff * j_norm_mean.item()
    plr = torch.square(j_norm - target_scale)
    pl_reg = plr * plr_factor
    return torch.mean(pl_reg), target_scale

# This is color matching loss, see Eq. 4
# It takes histogram of generated and target
def hellinger_dist_loss(g, hist_t, device):
    relu_g = torch.nn.functional.relu(g, inplace=False)
    relu_g = torch.clamp(relu_g, 0, 1) # Fix relu inplace gradient
    hist_g = utils.histogram_feature_v2(relu_g, device=device)  # Compute histogram feature of generated img
    t_sqred = torch.sqrt(hist_t)
    # print("Target", torch.isnan(t_sqred))
    g_sqred = torch.sqrt(hist_g)
    # print("Gen", hist_g)
    diff = t_sqred - g_sqred
    h = torch.sum(torch.square(diff), dim=(1,2,3))
    # print(hist_t.min(), hist_g.min())
    h_norm = torch.sqrt(h)
    h_norm = h_norm * (torch.sqrt(torch.ones((g.shape[0]))/2).to(device))
    
    # Used mean reduction, other option is sum reduction
    h_norm = torch.mean(h_norm)

    return h_norm