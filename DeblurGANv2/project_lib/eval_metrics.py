import torch
import math
from math import log10, sqrt
import numpy as np
import torch.nn.functional as F
import cv2


def PSNR(deblurIm, sharpIm):
    """
    Calculates the PSNR of the deblurred image
    using the formula:
    PSNR = 20* log10(maxI/sqrt(mean(delburred image- target image)**2))
    """
    deblurIm = deblurIm.astype(np.float64)
    sharpIm = sharpIm.astype(np.float64)
    mse = np.mean((deblurIm - sharpIm) ** 2)
    # MSE is zero means no noise is present in the signal .
    # Therefore PSNR have no importance.
    if(mse == 0):  
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    
    return psnr


def SSIM(deblurIm, sharpIm):
    """
    Calculates the SSIM (The structural similarity index measure) between
    the target image and the deblurred image with:
    
    SSIM = mean((2*mu_d*mu_s + C1)(2*sig_d_sig_s + C2)/(mu_d^2 + mu_s^2 + C1)(sig_d + sig_s + C2))
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    deblurIm = deblurIm.astype(np.float64)
    sharpIm = sharpIm.astype(np.float64)
    
    filt_ = cv2.getGaussianKernel(11, 1.5)
    filt = np.outer(filt_, filt_.transpose())

    mu_d = cv2.filter2D(deblurIm, -1, filt)[5:-5, 5:-5]
    mu_s = cv2.filter2D(sharpIm, -1, filt)[5:-5, 5:-5]
    
    mu_d_2 = mu_d**2
    mu_s_2 = mu_s**2
    mu_d_mu_s = mu_d * mu_s
    
    sig_d = cv2.filter2D(deblurIm**2, -1, filt)[5:-5, 5:-5] - mu_d_2
    sig_s = cv2.filter2D(sharpIm**2, -1, filt)[5:-5, 5:-5] - mu_s_2
    sig_d_sig_s = cv2.filter2D(deblurIm * sharpIm, -1, filt)[5:-5, 5:-5] - mu_d_mu_s

    ssim_ = ((2 * mu_d_mu_s + C1) * (2 * sig_d_sig_s + C2)) / ((mu_d_2 + mu_s_2 + C1) * (sig_d + sig_s + C2))
    
    ssim = ssim_.mean()
    
    return ssim

