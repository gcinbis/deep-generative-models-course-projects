# Loss Section
import torch
import torch.nn.functional as F
from torch.autograd import grad


# L1 norm implementation using linalg.norm
def l1_norm(x):
    return torch.mean(torch.abs(x))


# L2 norm implementation using linalg.norm
def l2_norm(x):
    # return torch.linalg.norm(x, 2, -1)
    return torch.sqrt(torch.sum(x ** 2, dim=1))


# Adversarial Loss of Discriminator
def adv_discr_loss(discr, x, x_gen, gen_class, real_class, lamb, device="cpu"):
    # t is used to calculate x_hat which is used to calculate grad_penalty
    # t is randomized between 0 and 1
    t = torch.rand(x.size()[0], 1, 1, 1).to(device)
    #t = t.expand_as(x)
    #t = t.to(device)
    # x_hat random noise is calculated as following formula, this formula is presented on
    # the paper Improved Training of Wasserstein GANs which explains steps to
    # create a WGAN with gradient penalty
    # With this approach, x_hat is sampled along lines between x_gen and x
    x_hat = t * x + (1 - t) * x_gen
    # x_hat.requires_grad = True
    x_hat = x_hat.to(device)
    # Attributes and classes are assumed to be returned as log of actual attributes and classes
    x_hat_class, x_hat_attr = discr(x_hat)
    x_hat_attr = x_hat_attr.to(device)
    x_hat_class = x_hat_class.to(device)

    # Gradient of gradient is taken by using autograd.grad
    # inputs : x_hat
    # outputs : classes or labels our discriminator predicted for given x_hat values
    # grad_outputs : Vector in the Jacobian-Vector product, needed in order to calculate higher order derivatives
    # create_graph : It is set as true to be able to compute higher order derivative products
    # retain_graph : To be able to loss.backward with higher order derivatives, it is needed to be set as true
    grads = grad(
        outputs=x_hat_class,
        inputs=x_hat,
        grad_outputs=torch.ones(x_hat_class.size()).to(device),
        create_graph=True,
        retain_graph=True)[0]

    grads = grads.view(grads.shape[0], -1)

    # L2 norm of gradients are taken and gradient formula is followed
    gradient_penalty = lamb * torch.mean((l2_norm(grads) - 1) ** 2)

    # Adversarial loss is returned by following calculation
    return torch.mean(gen_class) - torch.mean(real_class) + gradient_penalty


# Attribute Manipulation Loss of Discriminator
# Attribute manipulation is checked between predicted real attributed by discriminator and 
# source attributes which are the original attributes of the real input image.
def discr_attr_manip_loss(real_attr, attr_src):
    # To make 1 - torch.exp(real_attr) non zero, it is passed to sigmoid function and 
    # clamped with a previously selected epsilon number which is very small in order to not disturb the loss function itself
    # return -1 * torch.mean(attr_src * real_attr + (1 - attr_src) * F.logsigmoid(1 - torch.exp(real_attr)))
    real_attr = real_attr.float()
    attr_src = attr_src.float()
    return F.binary_cross_entropy_with_logits(real_attr, attr_src, reduction='sum') / real_attr.size(0)


# Model Objective of Discriminator
# Takes Discriminator, Generator, Real Images, Attribute differences, Target attributes, 
#       lambda value for gradient_penalty constant,
#       lambda1 model tradeoff param from the formulate of model objective of Discriminator,
#       and device for x_hat computations on gradient_penalty
# Returns the model objective loss of Discriminator
def discr_model_obj(discr, gen, x, attr_src, attr_targ, lamb, lamb1, device="cpu"):
    # Difference of attributes are calculated.
    attr_diff = attr_targ - attr_src
    # attr_diff = attr_diff * torch.rand_like(attr_diff)

    # New images are generated for given x and attr_diff
    x_gen = gen(x, attr_diff)
    #x_gen = x_gen.to(device)

    # Corresponding class and attributes are classified by discrimiator for generated image
    # Attributes and classes are assumed to be returned as log of actual attributes and classes
    gen_class, gen_attr = discr(x_gen.detach())
    #gen_attr = gen_attr.to(device)
    #gen_class = gen_class.to(device)
    # Corresponding class and attributes are classified by discrimiator for real image x
    # Attributes and classes are assumed to be returned as log of actual attributes and classes
    real_class, real_attr = discr(x)
    #real_attr = real_attr.to(device)
    #real_class = real_class.to(device)

    return -1 * adv_discr_loss(discr, x, x_gen, gen_class, real_class, lamb, device) + lamb1 * discr_attr_manip_loss(
        real_attr, attr_src)


# Reconstruction Loss of Generator
# Takes attribute difference as 0 and reconstructs the given x image and returns l1_norm 
# between reconstructions and original images.
# To represent attribute difference vector, tensor filled with zeros is created.
# The attribute difference tensors length is 13 to match the length of every other attribute tensors.
def reconst_loss(gen, x, device="cpu"):
    batch_size = x.shape[0]
    x_recons = gen(x, torch.zeros(batch_size, 13).to(device))
    x_recons = x_recons.to(device)
    return l1_norm(x - x_recons)


# Adverserial Loss of Generator
# Takes classes predicted by Discriminator for generated images
# Returns the adverserial loss which is mean of them
def adv_gen_loss(gen_class):
    return torch.mean(gen_class)


# Attribute Manipulation Loss of Generator
# Takes the attributes predicted by Discriminator for generated images and target attributes
# Returns the attribute manip loss of generator
def gen_attr_manip_loss(gen_attr, attr_targ):
    # inner_log = torch.sigmoid(1 - torch.exp(gen_attr))
    # inner_log = torch.clamp(inner_log, epsilon, 1. - epsilon)
    # return -1 * torch.mean(attr_targ * gen_attr + (1 - attr_targ) * F.logsigmoid(1 - torch.exp(gen_attr))) 
    # attr_targ_expanded = attr_targ.repeat(gen_attr.size(0), 1)
    # return F.binary_cross_entropy_with_logits(gen_attr, attr_targ_expanded, reduction='sum') / gen_attr.size(0)
    gen_attr = gen_attr.float()
    attr_targ = attr_targ.float()
    return F.binary_cross_entropy_with_logits(gen_attr, attr_targ, reduction='sum') / gen_attr.size(0)


# Model Objective of Generator
# Takes Discriminator, Generator, Real Images, Attribute differences, Target attributes, 
#       lambda2 and lambda3 model tradeoff params from the formulate of model objective of Generator,
#       and device if needed
def gen_model_obj(discr, gen, x, attr_src, attr_targ, lamb2, lamb3, device="cpu"):
    # Difference of attributes are calculated.
    attr_diff = attr_targ - attr_src
    # attr_diff = attr_diff * torch.rand_like(attr_diff)  # ?

    # New images are generated for given x and attr_diff
    x_gen = gen(x, attr_diff)
    x_gen = x_gen.to(device)

    # Corresponding class and attributes are classified by discrimiator for generated image
    # Attributes and classes are assumed to be returned as log of actual attributes and classes
    gen_class, gen_attr = discr(x_gen)
    gen_attr = gen_attr.to(device)
    gen_class = gen_class.to(device)

    return -1 * adv_gen_loss(gen_class) + lamb2 * gen_attr_manip_loss(gen_attr, attr_targ) + lamb3 * reconst_loss(gen,
                                                                                                                  x,
                                                                                                                  device)
