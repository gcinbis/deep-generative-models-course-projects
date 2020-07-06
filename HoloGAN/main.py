"""
HoloGAN implementation in PyTorch
May 17, 2020
"""
import torch
from init import initializer
from hologan import HoloGAN

def main():
    """Main functionsss"""
    args = initializer()
    torch.cuda.manual_seed_all(args.seed)
    model = HoloGAN(args)
    if not args.sampling:
        model.train(args)
    model.sample(args, trained=True, collection=True)

if __name__ == "__main__":
    main()
