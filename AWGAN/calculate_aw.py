import torch
import torch.nn.functional as F
import copy

def get_aw(real_loss, fake_loss, discriminator, discriminator_optim, real_scores, fake_scores):
    # These are straight from Algorithm 1 of paper
    alpha1 = 0.5
    alpha2 = 0.75
    epsilon = 0.05
    delta = 0.05
    dummy = 1e-5

    sr = F.sigmoid(real_scores).mean()
    sf = F.sigmoid(fake_scores).mean()

    real_loss.backward()
    real_grads = {}
    flat_real_grads = []
    for key, val in discriminator.named_parameters():
        if val.grad is not None:
            real_grads[key] = copy.deepcopy(val.grad)
            flat_real_grads.append(real_grads[key].view(-1)) 
            #flat_real_grads.append(torch.flatten(real_grads[key])) Flatten works terrible somehow !
            device = val.grad.device

    flat_real_grads = torch.cat(flat_real_grads)
    real_dot = torch.dot(flat_real_grads, flat_real_grads) + dummy
    real_norm = real_dot ** 0.5

    discriminator_optim.zero_grad()
    fake_loss.backward()
    fake_grads = {}
    flat_fake_grads = []
    for key, val in discriminator.named_parameters():
        if val.grad is not None:
            fake_grads[key] = copy.deepcopy(val.grad)
            flat_fake_grads.append(fake_grads[key].view(-1))
            #flat_fake_grads.append(torch.flatten(fake_grads[key]))
    
    flat_fake_grads = torch.cat(flat_fake_grads)
    fake_dot = torch.dot(flat_fake_grads, flat_fake_grads) + dummy
    fake_norm = fake_dot ** 0.5

    real_fake_dot = torch.dot(flat_real_grads, flat_fake_grads)
    # Unnormalized adaptive weight Algorithm 1
    if (sr < sf - delta) or sr < alpha1:
        if real_fake_dot < 0.0:
            wr = (1/real_norm) + epsilon
            wf = (-real_fake_dot / (fake_dot * real_norm)) + epsilon
        else:
            wr = (1/real_norm) + epsilon
            wf = epsilon

    elif (sr > sf - delta) and (sr > alpha2):
        if real_fake_dot < 0.0:
            wr = (-real_fake_dot /(real_dot * fake_norm)) + epsilon
            wf = (1/fake_norm) + epsilon
        else:
            wr = epsilon
            wf = (1/fake_norm) + epsilon

    else:
        wr = (1/real_norm) + epsilon
        wf = (1/fake_norm) + epsilon

    loss = wr * real_loss + wf * fake_loss

    for key, val in discriminator.named_parameters():
        if val.grad is not None:
            val.grad = wr * real_grads[key] + wf * fake_grads[key]

    return loss