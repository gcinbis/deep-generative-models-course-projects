import os

import torch
import torch.nn as nn

from utils import exact_interpolate, np_image_to_normed_tensor, normed_tensor_to_np_image, create_scale_pyramid


class Conv2DBlock(nn.Module):
    """ Combine Conv2d-BN-LReLU into a single block """

    # the 0.2 negative slope is given in the supplementary materials
    def __init__(self, in_channels, out_channels, kernel_size,  # conv arguments
                 use_bn=True, activation=None,  # customization of following blocks
                 conv_kwargs=None, bn_kwargs=None):  # optional kwargs for conv and bn

        # mutable default arguments are dangerous
        if conv_kwargs is None:
            conv_kwargs = {}
        if bn_kwargs is None:
            bn_kwargs = {}

        # call superclass init and (maybe) create layers
        super().__init__()
        if bn_kwargs is None:
            bn_kwargs = {}
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **conv_kwargs)
        self.bn = nn.BatchNorm2d(out_channels, **bn_kwargs) if use_bn else nn.Identity()
        self.activ = activation if activation else nn.Identity()

    def forward(self, x):
        return self.activ(self.bn(self.conv(x)))


class SGNet(nn.Module):
    """
    A class to create the networks used in the SinGAN paper. Each generator and 
    discriminator is very similar, being composed of 5 blocks of 
    (conv2d, batch_norm, leaky_relu) blocks, with the final one being slightly different.
    All intermediate blocks have the same amount of kernels and a kernel size of 3x3.
    Zero padding is done initially, so that the network preserves the shape of its input.
    """

    def __init__(self, num_blocks=5, kernel_count=32, kernel_size=3,  # architecture customization
                 final_activation=nn.Tanh(), final_bn=False,  # final layer cust.
                 input_channels=3, output_channels=3):  # channel counts

        # superclass init and add the initial padding layer
        super().__init__()
        layers = [nn.ZeroPad2d(num_blocks)]  # since kernel size is 3, pad 1 per block

        # loop to create each layer except last, 
        # all properties are shared except for the number of channels
        def sgnet_block(in_channels, out_channels):
            return Conv2DBlock(in_channels, out_channels, kernel_size,
                               activation=nn.LeakyReLU(negative_slope=0.2))  # as given in the paper

        layers.append(sgnet_block(input_channels, kernel_count))  # first layer
        for _ in range(num_blocks - 2):  # last layer has a different architecture
            layers.append(sgnet_block(kernel_count, kernel_count))
        # the final activation depends on whether this is the generator or critic
        # (tanh for gen. and none for crit.), and is different from the others
        final_block = Conv2DBlock(kernel_count, output_channels, kernel_size,
                                  final_bn, final_activation)
        layers.append(final_block)

        # create a sequential model from it
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # simply forwards through the layers
        return self.model(x)


class NoiseSampler:
    """
    This functor provides a common interface from which we draw the noise samplers,
    to make it easy to control all the sampling from one code block, we could easily
    change from normal to uniform just by changing one line here, for example.
    A noise sampler simply takes a reference tensor and produces noise with the same shape.
    Object rather than closure so that it can be pickled without python complaining.
    """
    def __init__(self, noise_std):
        self.noise_std = noise_std

    def __call__(self, x):
        return self.noise_std * torch.randn_like(x)


class SGGen(nn.Module):
    """
    This class adds the extra fluff (noise sampling and residual connections)
    on top of the basic SGNet architecture to create the full single-scale generator.
    """
    def __init__(self, sgnet, noise_std):
        super().__init__()
        self.sgnet = sgnet
        self.noise_sampler = NoiseSampler(noise_std)

    def forward(self, x, z=None):
        if z is None:
            z = self.noise_sampler(x)
        g_in = x + z  # image + noise as input
        g_out = self.sgnet(g_in) + x  # residual connection
        return g_out


class MultiScaleSGGenView(nn.Module):
    """ 
    This class serves as a 'view' over the list of generators that makes the stack
    look like a single generator. Multiple scales of generators are combined by
    starting from the lowest; the output of the lower scale is resized and 
    passed to the upper generator automatically until no more are left. 
    In the end we have something that takes an image input and returns
    another image, just like a single generator. 
    
    Attributes:
        generators: a list of SGGen's representing generator networks, converted
            to nn.ModuleList when stored
        scaling_factor: a floating point scalar which represents the scale multiplier
            between each generator (e.g. 1.25)
        scaling_mode: a string for the scaling mode, should be a valid input for
            torch.nn.functional.interpolate's 

    Illustration of the full architecture:
            samplerN -> noiseN -> | generatorN | 
                        imgN-1 -> |            | -> imgN
                         ^
                         .............................
                         .......other generators......
                         .............................
                                                     ^
            sampler1 -> noise1 -> | generator1 |     |
                        img0   -> |            | -> img1
                         ^
                         |____________________________
                                                     ^
            sampler0 -> noise0 -> | generator0 |     |
                                  |            | -> img0

    Note about scaling:
        
        Simply using scaling_factor to scale outputs is nice when we do not
        have any strict requirements on image shapes, but does not really
        work when we expect a certain size for each output. Consider
        starting from a size of 250 and scaling by a factor of 3/4:
        
        scales = [250, 188, 141, 105, 79, 59, 44, 33, 25]

        Since we round the result at each step, the final output is 25, although
        250 * 0.75^8 ~ 25.08. If we take an input with size 25 and scale up with
        a factor 4/3 we get the following:

        scales = [25, 33, 44, 59, 79, 105, 140, 187, 250]

        Notice that some scales do not match because we started with 25 instead of
        25.08. This can be a problem when calculating reconstruction loss, for
        example. Thus, we provide an optional argument to the forward pass, a
        (float, float) tuple for providing the exact size (e.g. (25.08, 25.08) 
        rather than (25, 25) to be used when upsampling) to ensure that we obtain
        exact shape matches.

    """

    def __init__(self, generators, scaling_factor, scaling_mode='bicubic'):

        # initialize superclass and check arguments
        super().__init__()

        # assign members, nn.ModuleList for generators to ensure
        # proper behavior, e.g. .parameters() returning correctly
        self.generators = nn.ModuleList(generators)
        self.scaling_factor = scaling_factor
        self.scaling_mode = scaling_mode

        # freeze all generators except for the top one 
        for g in self.generators[:-1]:
            g.requires_grad_(False)
            g.eval()

    def forward(self, x, exact_size=None, z_input=None):
        """
        Forward pass through the network.

        Args: 
        x: a 4D (N, C, H, W) tensor input to the first (coarsest scale) generator,
        z_input: a list of 4D noise tensors to be used as the noise input at each scale,
            if None, the noise samplers are used to generate noise
        exact_size: a (float, float) tuple for providing the theoretical shape of the input,
            see the 'Note about scaling:' in the class docstring.
            if None, the size of x is used as the exact_size
        """
        # set exact_size as the input size if not provided
        if exact_size is None:
            exact_size = tuple(float(d) for d in x.shape[2:4])  # (H, W)

        # go through each generator
        x_out = None
        for i, g, in enumerate(self.generators):
            z = None if z_input is None else z_input[i]  # get the noise input from the proper source
            x_out = g(x, z)  # pass through
            if i < len(self.generators) - 1:  # upsample if not the last layer
                # interpolate using the exact dimensions and update them
                x, exact_size = exact_interpolate(x_out, self.scaling_factor, exact_size, self.scaling_mode)
        return x_out


class FixedInputSGGenView(nn.Module):
    """
    A wrapper to fix the input of an SGNet view for easier calls to forward, so that
    we do not have to provide the coarsest zero (or original image) input and exact size at each call
    """
    def __init__(self, sgnet_view, coarsest_input, coarsest_exact_size=None):
        super().__init__()
        if coarsest_exact_size is None:
            coarsest_exact_size = tuple(float(d) for d in coarsest_input.shape[2:4])
        self.sgnet_view = sgnet_view
        self.coarsest_exact_size = coarsest_exact_size
        self.coarsest_input = coarsest_input

    def forward(self, z_input=None, num_samples=1):
        # cool, but a large num_samples can eat up a lot of memory,
        # so we do not use num_samples > 1 in the notebook
        inputs = self.coarsest_input.expand(num_samples, -1, -1, -1)
        return self.sgnet_view.forward(inputs, self.coarsest_exact_size, z_input)


def save_model(model_path, image, generators, critics, upsampling_factor, upsampling_mode, downsampling_mode):
    """
    A function to save a trained model to the given path.

    Args:
        model_path: str, path to save the model to
        image: original image used to train, as a [-1, 1] torch tensor
        generators: list of trained SGGen generators
        critics: list of trained SGNet critics
        upsampling_factor: float, scaling factor used when training the model
        upsampling_mode: str, mode used when upsampling generator outputs (e.g. bilinear)
        downsampling_mode: str, mode used when downsampling the original image (e.g. bicubic)
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    # TODO: could change to encode image into png bytes rather than raw uint8
    image = normed_tensor_to_np_image(image)
    torch.save({
        'image': image,
        'generators': generators,
        'critics': critics,
        'upsampling_factor': upsampling_factor,
        'upsampling_mode': upsampling_mode,
        'downsampling_mode': downsampling_mode
    }, model_path)


def load_generator(model_path, input_scale=0, output_size=None, custom_input=False, inference=True, device='cpu'):
    """
    A function to load a saved model from disk and create a generator stack from it.

    Args:
        model_path: path to the saved model file, under
        input_scale: the scale from which the stack of generators will get their inputs
        output_size: a (float, float) tuple, output image size of the generator; if None, the default
            size with which the network was trained is used. Otherwise, a noise input in accordance
            with the output size is generated. Only compatible with input_scale == 0 and custom_input == False!
        custom_input: if specified, the function will return an MultiScaleSGGenView with no fixed inputs
            otherwise, a FixedInputSGGen view with zero or scaled original image input is returned
        inference: if False, the uppermost generator will have grads and be in training mode,
            otherwise, all the generators will be frozen and in eval mode
        device: device on which to load the model

    Returns:
        ms_gen: the multi-scale generator built with the given arguments in inference mode
        image: the original image the model was trained with, as a [-1, 1] torch tensor
    """
    if output_size is not None and (input_scale != 0 or custom_input):
        raise ValueError('output_size can only be set with input_scale == 0 and custom_input == False!')

    save_dict = torch.load(model_path, map_location=device)
    # build the view first
    ms_gen = MultiScaleSGGenView(save_dict['generators'][input_scale:],
                                 save_dict['upsampling_factor'], save_dict['upsampling_mode'])
    input_img = np_image_to_normed_tensor(save_dict['image']).to(device)
    if not custom_input:
        # create the scale pyramid
        num_scales = len(save_dict['generators'])
        downsampling_factor = 1.0 / save_dict['upsampling_factor']
        img_scales, scale_sizes = create_scale_pyramid(input_img, downsampling_factor,
                                                       num_scales, save_dict['downsampling_mode'])

        # default assignment for the input, valid when output_size is None and input_scale != 0
        i = -(input_scale + 1)
        ms_gen_input = img_scales[i]  # scaled original image as input
        if output_size is None:  # same as in training
            exact_input_size = scale_sizes[i]
        else:  # custom output size, calculate custom input size
            exact_input_size = tuple(float(d) * downsampling_factor**(num_scales-1) for d in output_size)
            # input scale is guaranteed to be zero here, so the input is set right afterwards
        if input_scale == 0:  # special case, give zeros as input for the coarsest scale
            rounded_input_size = tuple(round(d) for d in exact_input_size)
            ms_gen_input = torch.zeros(*input_img.shape[:2], *rounded_input_size, device=device)

        ms_gen = FixedInputSGGenView(ms_gen, ms_gen_input, exact_input_size)
    # inference mode
    if inference:
        ms_gen.requires_grad_(False)
        ms_gen.eval()
    return ms_gen, input_img
