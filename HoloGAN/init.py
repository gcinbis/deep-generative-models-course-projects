"""
HoloGAN implementation in PyTorch
May 17, 2020
"""
import argparse

def initializer():
    """initializer of the program.

    This parses and extracts all training and testing settings.
    """
    #pylint: disable=C0326, C0330
    parser = argparse.ArgumentParser(description="PyTorch HoloGAN implementation")
    parser.add_argument("--seed",               type=int, default=23, metavar="N",
                                                help="random seed")
    parser.add_argument("--image-path",         type=str, default="../dataset/celebA/",metavar="S",
                                                help="training dataset directory path (default: \'../dataset/celebA/\')")
    parser.add_argument("--dataset",            type=str, default="celebA", choices=["celebA"],
                                                help="dataset selection (default: celebA)")
    parser.add_argument("--gpu",                action="store_true", default=False,
                                                help="flag to enable cuda computation (default: False)")
    parser.add_argument("--batch-size",         type=int, default=32, metavar="N",
                                                help="training batch size of the model (default: 32)")
    parser.add_argument("--max-epochs",         type=int, default=50, metavar="N",
                                                help="the maximum number of epochs for training (default: 50)")
    parser.add_argument("--epoch-step",         type=int, default=25, metavar="N",
                                                help="epoch step to compute the adaptive learning rate (default: 25)")
    parser.add_argument("--z-dim",              type=int, default=128, metavar="N",
                                                help="the length of the generative model input (default: 128)")
    parser.add_argument("--d-lr",               type=float, default=0.0001, metavar="N",
                                                help="the learning rate of the discriminator (default: 0.0001)")
    parser.add_argument("--g-lr",               type=float, default=0.0001, metavar="N",
                                                help="the learning rate of the generator (default: 0.0001)")
    parser.add_argument("--beta1",              type=float, default=0.5, metavar="N",
                                                help="minimum betas parameter of the Adam optimizer (default: 0.5)")
    parser.add_argument("--beta2",              type=float, default=0.999, metavar="N",
                                                help="maximum betas parameter of the Adam optimizer (default: 0.999)")
    parser.add_argument("--lambda-latent",      type=float, default=0.0, metavar="N",
                                                help="the lambda latent coefficient given in the paper (default: 0.0)")
    parser.add_argument("--elevation-low",      type=int, default=0, metavar="N",
                                                help="the minimum elevation angle (default: 70)")
    parser.add_argument("--elevation-high",     type=int, default=0, metavar="N",
                                                help="the maximum elevation angle (default: 110)")
    parser.add_argument("--azimuth-low",        type=int, default=25, metavar="N",
                                                help="the minimum azimuth angle (default: 220)")
    parser.add_argument("--azimuth-high",       type=int, default=65, metavar="N",
                                                help="the maximum azimuth angle (default: 320)")
    parser.add_argument("--scale-low",          type=float, default=1.0, metavar="N",
                                                help="the minimum scaling value of 3D transformation (default: 1.0)")
    parser.add_argument("--scale-high",         type=float, default=1.0, metavar="N",
                                                help="the maximum scaling value of 3D transformation (default: 1.0)")
    parser.add_argument("--transX-low",         type=int, default=0, metavar="N",
                                                help="the minimum translation factor across the X-axis (default: 0)")
    parser.add_argument("--transX-high",        type=int, default=0, metavar="N",
                                                help="the maximum translation factor across the X-axis (default: 0)")
    parser.add_argument("--transY-low",         type=int, default=0, metavar="N",
                                                help="the minimum translation factor across the Y-axis (default: 0)")
    parser.add_argument("--transY-high",        type=int, default=0, metavar="N",
                                                help="the maximum translation factor across the Y-axis (default: 0)")
    parser.add_argument("--transZ-low",         type=int, default=0, metavar="N",
                                                help="the minimum translation factor across the Z-axis (default: 0)")
    parser.add_argument("--transZ-high",        type=int, default=0, metavar="N",
                                                help="the maximum translation factor across the Z-axis (default: 0)")
    parser.add_argument("--log-interval",       type=int, default=1000, metavar="N",
                                                help="logging interval in terms of batch size (default: 1000)")
    parser.add_argument("--update-g-every-d",   type=int, default=5, metavar="N",
                                                help="do not save the current model")
    parser.add_argument("--no-save-model",      action="store_true", default=False,
                                                help="flag to not save the current model (default: False)")
    parser.add_argument("--rotate-elevation",   action="store_true", default=False,
                                                help="rotate the z sampling with elevation (default: False)")
    parser.add_argument("--rotate-azimuth",     action="store_true", default=False,
                                                help="rotate the z sampling with azimuth (default: False)")
    parser.add_argument("--load-dis",           type=str, default=None, metavar="S",
                                                help="the path for loading and/or evaluating the discriminator")
    parser.add_argument("--load-gen",           type=str, default=None, metavar="S",
                                                help="the path for loading and/or evaluating the generator")
    parser.add_argument("--sampling",           action="store_true", default=False,
                                                help="enable the sampling mode (default: False)")
    parser.add_argument("--device",             help=argparse.SUPPRESS)
    parser.add_argument("--start-epoch",        help=argparse.SUPPRESS)
    parser.add_argument("--recorder",           help=argparse.SUPPRESS)
    parser.add_argument("--results-dir",        help=argparse.SUPPRESS)
    parser.add_argument("--models-dir",         help=argparse.SUPPRESS)
    parser.add_argument("--samples-dir",        help=argparse.SUPPRESS)
    parser.add_argument("--hist-file",          help=argparse.SUPPRESS)
    #pylint: enable=C0326, C0330
    return parser.parse_args()
