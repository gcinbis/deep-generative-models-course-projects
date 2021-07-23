# Experiments

These user study experiments were done to incrementally study components of the paper
implementation. To run each one, you should copy/move the notebook one level above this directory.

Disclaimer: Older notebooks may not work "out of the box"

In order, they progressed as follows:

1. dcgan_64x64.ipynb => Can we implement a standard DCGAN on a reduced version of our dataset?
2. dcgan_128x128.ipynb => Can we learn by extending the previoud model to the original resolution?
3. udcgan_64x64_with_cutmix.ipynb => Can we learn with a U-Net-like discriminator, using the loss
   that we can define in the paper? Furthermore, can we add CutMix examples and consistency loss?
4. udcgan_128x128_with_cutmix.ipynb => Can we train a 128x128 U-GAN-like network with all the tools in the
   original paper?
5. fid.ipynb => Can we compute statistics of learned models? How are PyTorch and TensorFlow FID different?
