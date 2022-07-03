import torch
from torch import nn
import numpy as np


class Hardmap_Mask_Update(nn.Module):
    """
    Class for calculation of hard map mask coefficients considering the distance to gap boundaries
    """

    def __init__(self, k_for_hard_map):
        """
        Initialize the hard map mask coefficients update
        """
        super().__init__()
        # Regional base coefficient for mask multiplier
        self.k = k_for_hard_map  # k value in Figure 3

    def forward(self, batch_size, mask):
        """
        e.g input : Initial hard map from mask input     e.g output: hard map with filled coefficients for a kernel size of 3
        M=torch.Tensor([[1,1,1,1,1],                         tensor([[1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
                        [1,0,0,0,0],                                 [1.0000, 0.5000, 0.5000, 0.5000, 0.5000],
                        [1,0,0,0,0],                                 [1.0000, 0.5000, 0.5000, 0.5000, 0.5000],
                        [1,0,0,0,0],                                 [1.0000, 0.5000, 0.5000, 0.2500, 0.2500],
                        [0,0,0,0,0]])                                [0.5000, 0.5000, 0.5000, 0.2500, 0.2500]])
        """

        for imask in range(batch_size):
            for idlt in range(2):  # Shore_lines, Dilation loop, if the filled area is large set it to a high value

                # Check batch size compatible with mask shape
                if imask >= mask.shape[0]:
                    break

                # Only search for mask kernels around empty pixels
                M_indices_for_zero_elements = np.argwhere(mask[imask].cpu().detach() == 0)

                # Check if are there any zero pixels if so continue padding the mask
                if len(M_indices_for_zero_elements[0]) > 0:
                    pass
                else:
                    break

                    # Add padding to the mask so that kernel will be generic also for the picture edges.
                M_pad = torch.nn.functional.pad(mask[imask], pad=(1, 1, 1, 1), mode='constant', value=0)

                # Initiate the 3x3 kernel and storage tensors for indices
                M_kernel = torch.zeros(len(M_indices_for_zero_elements[0]), 3, 3)
                Index_j_kernel = torch.zeros(len(M_indices_for_zero_elements[0]), 3, 3)
                Index_k_kernel = torch.zeros(len(M_indices_for_zero_elements[0]), 3, 3)

                # Calculate kernels around zero elements
                for i in range(len(M_indices_for_zero_elements[0])):
                    idx = M_indices_for_zero_elements[0][i].item()
                    idy = M_indices_for_zero_elements[1][i].item()
                    for j in range(3):
                        for k in range(3):
                            M_kernel[i, j, k] = M_pad[idx + j, idy + k]
                            Index_j_kernel[i, j, k] = idx + j
                            Index_k_kernel[i, j, k] = idy + k

                # Filter the kernels which have element summation greater than 0
                get_ith_index = np.argwhere(torch.sum(M_kernel, (1, 2)) > 0)
                M_kernel_nb = torch.flatten(M_kernel[get_ith_index])
                Index_j_kernel_nb = torch.flatten(Index_j_kernel[get_ith_index])
                Index_k_kernel_nb = torch.flatten(Index_k_kernel[get_ith_index])

                # Get indices of zero pixels which have kernel summation greater than 0
                # which means that it is a neighbour kernel to the boundary
                get_2nd_index = np.argwhere(M_kernel_nb == 0)
                Index_j_kernel_el = Index_j_kernel_nb[get_2nd_index]
                Index_k_kernel_el = Index_k_kernel_nb[get_2nd_index]
                node_indices = torch.concat((Index_j_kernel_el, Index_k_kernel_el), 1)

                # Remove duplicates
                node_indices = torch.unique(node_indices, dim=0)
                node_indices = node_indices.long()

                # Apply coefficient on padded mask
                M_pad[node_indices[:, 0], node_indices[:, 1]] = 1 / self.k ** (idlt + 1)

                # Crop the padded mask and set for mask
                Mnew = M_pad[1:-1, 1:-1]
                mask[imask] = Mnew
            # save random hard map just for inspection
            # save_image(M[imask], 'hard_mask' + '_' + str(imask) + '_'+
            #     str(M[imask].shape) + '_' + str(np.random.rand())+'.png')
        return mask


class HardSPDNorm(nn.Module):
    """
    Hard SPD Normalization block
    """

    def __init__(self, channels, img_size, k_for_hard_map):
        """
        Initialize the Hard SPD Normalization block
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm([channels, img_size, img_size])  # Normalization layer for f_in
        self.img_size = img_size
        self.k_for_hard_map = k_for_hard_map
        self.mask_update = Hardmap_Mask_Update(k_for_hard_map=k_for_hard_map)

        self.first_layer = nn.Sequential(  # First convolution&activation layer for the image and the mask
            nn.Conv2d(3, 4, 3, 1, 1),
            nn.ReLU()
        )

        self.gamma_layer = nn.Conv2d(4, channels, 3, 1, 1)  # Convolution for gamma map
        self.beta_layer = nn.Conv2d(4, channels, 3, 1, 1)  # Convolution for beta map

    def forward(self, f_in, mask, img):
        """
        Calculate Hard SPDNorm
        :param f_in: input features
        :param mask: mask image
        :param img: pc-conv image
        :return: output features
        """
        normalized_fin = self.layer_norm(f_in)

        conv_img = self.first_layer(img)
        gamma = self.gamma_layer(conv_img)
        beta = self.beta_layer(conv_img)

        # Hard Spectral Normalization
        M = mask.clone()
        M[mask > 0] = 1
        # Since hard map calculation takes too much time close it for the last normalization
        if self.img_size < 128:
            # Calculation of hard map masks prior to hard spectral normalization layers
            M = torch.squeeze(M, 1)
            batch_size = f_in.shape[0]
            mask_hard = self.mask_update(batch_size, M)
            mask_hard = torch.unsqueeze(mask_hard, 1)
        else:
            mask_hard = M

        f_out = normalized_fin * (gamma * mask_hard.clone()) + (beta * mask_hard.clone())
        return f_out


class SoftSPDNorm(nn.Module):
    """
    Soft SPD Normalization block
    """

    def __init__(self, channels, img_size):
        """
        Initialize the Soft SPD Normalization block
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm([channels, img_size, img_size])  # Normalization layer for f_in
        self.first_layer = nn.Sequential(  # First convolution&activation layer for the image and the mask
            nn.Conv2d(3, 4, 3, 1, 1),
            nn.ReLU()
        )

        self.gamma_layer = nn.Conv2d(4, channels, 3, 1, 1)  # Convolution for gamma map
        self.beta_layer = nn.Conv2d(4, channels, 3, 1, 1)  # Convolution for beta map

        self.conv_fp = nn.Conv2d(3, 10, 3, 1, 1)  # Convolution for Fp
        self.conv_fp_fin = nn.Conv2d(10 + channels, channels, 3, 1, 1)  # Convolution for Fp+Fin
        self.sigmoid = nn.Sigmoid()

    def forward(self, f_in, mask, img):
        """
        Calculate Hard SPDNorm
        :param f_in: input features
        :param mask: mask image
        :param img: pc-conv image
        :return: output features
        """
        normalized_fin = self.layer_norm(f_in)

        conv_img = self.first_layer(img)
        gamma = self.gamma_layer(conv_img)
        beta = self.beta_layer(conv_img)

        # calculate soft diverse map
        M = mask.clone()
        M[mask > 0] = 1
        Fp = self.conv_fp(img)

        concat = torch.concat([Fp, f_in], 1)
        out2 = self.conv_fp_fin(concat)
        out3 = self.sigmoid(out2)
        d_soft = torch.mul(out3, (1 - M)) + M

        # save random soft map just for inspection
        # save_image(d_soft[0,5,:,:], 'soft_mask' + '_'+ str(d_soft.shape)
        #            + '_' + str(np.random.rand())+'.png')

        # apply scale and bias
        out = normalized_fin * (d_soft * gamma) + (beta * d_soft)
        return out


class SPD_Norm_Residual_Block(nn.Module):
    """ A class implementing Residual Blocks of SPD Normalization"""

    def __init__(self, channels, img_size, k_for_hard_map):
        """
        Initialize the Residual SPD Normalization Block
        """
        super().__init__()

        self.hsn1 = HardSPDNorm(channels, img_size, k_for_hard_map=k_for_hard_map)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)

        self.hsn2 = HardSPDNorm(channels, img_size, k_for_hard_map=k_for_hard_map)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

        self.ssn1 = SoftSPDNorm(channels, img_size)
        self.relu3 = nn.ReLU(inplace=False)
        self.conv3 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, f_in, mask, image):
        """
        Given f_in, scaled image and the scaled mask, calculate the f_out
        :param f_in: input features
        :param mask: mask image
        :param image: pc conv image
        :return: output features
        """

        # Hard Spectral Normalization
        out1 = self.hsn1(f_in, mask, image)
        out1 = self.relu1(out1)
        out1 = self.conv1(out1)

        out2 = self.hsn2(out1, mask, image)
        out2 = self.relu2(out2)
        out2 = self.conv2(out2)

        # Soft Spectral Normalization
        out3 = self.ssn1(f_in, mask, image)
        out3 = self.relu3(out3)
        out3 = self.conv3(out3)

        f_out = out3 + out2
        return f_out


class PDGAN(nn.Module):
    """ Probabilistic diverse PD-GAN class """

    def __init__(self, first_channel_size, img_size, k_for_hard_map, noise_size):
        """
        Initialize the PDGAN model architecture.
        """
        super().__init__()

        self.noise_size = int(noise_size)
        self.first_channel_size = int(first_channel_size)
        self.img_size = int(img_size)
        self.latent_vector_size = int(self.first_channel_size * self.img_size * self.img_size / 256)
        "Initial fully connecter layer"
        self.fc = nn.Linear(self.noise_size * self.noise_size,
                            self.latent_vector_size)

        "Pooling layers for downsample the image and the mask"
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.first_norm_channel_size = self.first_channel_size
        self.first_norm_img_size = int(self.img_size / 16)

        self.second_norm_channel_size = int(3 * self.first_channel_size / 4)
        self.second_norm_img_size = int(self.img_size / 8)

        self.third_norm_channel_size = int(self.first_channel_size / 2)
        self.third_norm_img_size = int(self.img_size / 4)

        self.last_norm_channel_size = int(self.first_channel_size / 4)
        self.last_norm_img_size = int(self.img_size / 2)

        "Normalization Blocks"
        self.sn1 = SPD_Norm_Residual_Block(channels=self.first_norm_channel_size,
                                           img_size=self.first_norm_img_size, k_for_hard_map=k_for_hard_map)

        self.sn2 = SPD_Norm_Residual_Block(channels=self.second_norm_channel_size,
                                           img_size=self.second_norm_img_size, k_for_hard_map=k_for_hard_map)

        self.sn3 = SPD_Norm_Residual_Block(channels=self.third_norm_channel_size,
                                           img_size=self.third_norm_img_size, k_for_hard_map=k_for_hard_map)

        self.sn4 = SPD_Norm_Residual_Block(channels=self.last_norm_channel_size,
                                           img_size=self.last_norm_img_size, k_for_hard_map=k_for_hard_map)

        "Deconvolution layers"
        self.deconv1 = nn.ConvTranspose2d(in_channels=self.first_norm_channel_size,
                                          out_channels=self.second_norm_channel_size, kernel_size=2, stride=2)

        self.deconv2 = nn.ConvTranspose2d(in_channels=self.second_norm_channel_size,
                                          out_channels=self.third_norm_channel_size, kernel_size=2, stride=2)

        self.deconv3 = nn.ConvTranspose2d(in_channels=self.third_norm_channel_size,
                                          out_channels=self.last_norm_channel_size, kernel_size=2, stride=2)

        self.deconv4 = nn.ConvTranspose2d(in_channels=self.last_norm_channel_size,
                                          out_channels=3, kernel_size=2, stride=2)

    def forward(self, img, mask, noise):
        """ Given a padded image and its mask, calculate the generated samples"""
        batch_size = img.shape[0]
        # Scale the image for different SPD normalization stages by downsampling

        img1 = self.pool1(img)  # 3x128x128
        img2 = self.pool2(img1)  # 3x64x64
        img3 = self.pool3(img2)  # 3x32x32
        img4 = self.pool4(img3)  # 3x16x16

        # Scale the mask for different SPD normalization stages by downsampling

        mask1 = self.pool1(mask)  # 1x128x128
        mask2 = self.pool2(mask1)  # 1x64x64
        mask3 = self.pool3(mask2)  # 1x32x32
        mask4 = self.pool4(mask3)  # 1x16x16

        # Generate 2D random Gaussian noise and reshape it to 1D
        noise = torch.reshape(noise, (batch_size, self.noise_size * self.noise_size))

        # First layer is a fully connected layer
        out = self.fc(noise)

        # Reshape 1D vector output to block output (channels=24,height=16,width=16)

        out = torch.reshape(out, (batch_size, self.first_norm_channel_size,
                                  self.first_norm_img_size, self.first_norm_img_size))

        # Perform forward pass with SPD normalization blocks followed by deconvolution
        out = self.sn1(out, mask4, img4)
        out = self.deconv1(out)
        out = self.sn2(out, mask3, img3)
        out = self.deconv2(out)
        out = self.sn3(out, mask2, img2)
        out = self.deconv3(out)
        out = self.sn4(out, mask1, img1)
        out = self.deconv4(out)
        return img, mask, out
