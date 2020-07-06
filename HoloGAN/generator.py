"""
HoloGAN Generator implementation in PyTorch
May 17, 2020
"""
import torch
from torch import nn
from torch.autograd import Variable

class ZMapping(nn.Module):
    def __init__(self, z_dimension, output_channel):
        super(ZMapping, self).__init__()
        self.output_channel = output_channel
        self.linear1 = nn.Linear(z_dimension, output_channel * 2)
        nn.init.normal_(self.linear1.weight, std=0.02)
        nn.init.constant_(self.linear1.bias, val=0.0)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.linear1(x))
        return out[:, :self.output_channel], out[:, self.output_channel:]

class BasicBlock(nn.Module):
    """Basic Block defition of the Generator.
    """
    def __init__(self, z_planes, in_planes, out_planes, transpose_dim):
        super(BasicBlock, self).__init__()
        if transpose_dim == 2:
            self.convTranspose = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4,
                                                    stride=2, padding=1)
        else:
            self.convTranspose = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=3,
                                                    stride=2, output_padding=1, padding=1)

        nn.init.normal_(self.convTranspose.weight, std=0.02)
        nn.init.constant_(self.convTranspose.bias, val=0.0)
        self.zMapping = ZMapping(z_planes, out_planes)
        self.relu = nn.ReLU()

    def forward(self, h, z):
        h = self.convTranspose(h)
        s, b = self.zMapping(z)
        h = AdaIn(h, s, b)
        h = self.relu(h)
        return h

class Generator(nn.Module):
    def __init__(self, in_planes, out_planes, z_planes, view_planes=6, gpu=True):
        super(Generator, self).__init__()
        self.device = torch.device("cuda" if gpu else "cpu")
        tensor = (torch.randn(1, in_planes*8, 4, 4, 4) - 0.5 ) / 0.5
        self.x = nn.Parameter(tensor.to(self.device))
        self.x.requires_grad = True

        self.zMapping = ZMapping(z_planes, in_planes*8)
        self.block1 = BasicBlock(z_planes, in_planes=in_planes*8, out_planes=in_planes*2,
                                 transpose_dim=3)
        self.block2 = BasicBlock(z_planes, in_planes=in_planes*2, out_planes=in_planes,
                                 transpose_dim=3)

        self.convTranspose2d1 = nn.ConvTranspose2d(in_planes*16, in_planes*16, kernel_size=1)
        nn.init.normal_(self.convTranspose2d1.weight, std=0.02)
        nn.init.constant_(self.convTranspose2d1.bias, val=0.0)

        self.block3 = BasicBlock(z_planes, in_planes=in_planes*16, out_planes=in_planes*4,
                                 transpose_dim=2)
        self.block4 = BasicBlock(z_planes, in_planes=in_planes*4, out_planes=in_planes,
                                 transpose_dim=2)

        self.convTranspose2d2 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, padding=1)
        nn.init.normal_(self.convTranspose2d2.weight, std=0.02)
        nn.init.constant_(self.convTranspose2d2.bias, val=0.0)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, z, view_in):

        batch_size = z.shape[0]
        x = self.x.repeat(batch_size, 1, 1, 1, 1)
        s0, b0 = self.zMapping(z)
        h0 = AdaIn(x, s0, b0)
        h0 = self.relu(h0)

        h1 = self.block1(h0, z)
        h2 = self.block2(h1, z)

        h2_rotated = self.transformation3d(h2, view_in, 16, 16)
        h2_rotated = h2_rotated.permute(0, 1, 3, 2, 4)
        inv_idx = torch.arange(h2_rotated.size(2)-1, -1, -1).long()
        h2_rotated = h2_rotated[:, :, inv_idx, :, :]
        h2_2d = h2_rotated.reshape(batch_size, -1, 16, 16)

        h3 = self.convTranspose2d1(h2_2d)
        h3 = self.relu(h3)

        h4 = self.block3(h3, z)
        h5 = self.block4(h4, z)

        h6 = self.convTranspose2d2(h5)
        h6 = self.tanh(h6)
        return h6

    def transformation3d(self, voxel_array, view_params, size=16, new_size=16):
        # TODO: daha efficient olması için ileri de tek bir matrix formatında oluşturulabilir

        theta = Variable(torch.as_tensor(view_params[:, 0].reshape(-1, 1, 1)).float(),
                         requires_grad=True).to(self.device)
        gamma = Variable(torch.as_tensor(view_params[:, 1].reshape(-1, 1, 1)).float(),
                         requires_grad=True).to(self.device)
        ones  = torch.ones(theta.shape, requires_grad=True).to(self.device)
        zeros = torch.zeros(theta.shape, requires_grad=True).to(self.device)

        # Rotation azimuth (i.e. rotate around-z)
        rot_z = torch.cat([
            torch.cat([theta.cos(),   theta.sin(),  zeros,  zeros], dim=2),
            torch.cat([-theta.sin(),  theta.cos(),  zeros,  zeros], dim=2),
            torch.cat([zeros,         zeros,        ones,   zeros], dim=2),
            torch.cat([zeros,         zeros,        zeros,  ones],  dim=2)], dim=1)

        # Rotation elevation (i.e. rotate around-x)
        rot_y = torch.cat([
            torch.cat([gamma.cos(),   zeros,  gamma.sin(),  zeros], dim=2),
            torch.cat([zeros,         ones,   zeros,        zeros], dim=2),
            torch.cat([-gamma.sin(),  zeros,  gamma.cos(),  zeros], dim=2),
            torch.cat([zeros,         zeros,  zeros,        ones],  dim=2)], dim=1)

        rotation_matrix = torch.matmul(rot_z, rot_y)

        # Scaling matrix
        scale = Variable(torch.as_tensor(view_params[:, 2].reshape(-1, 1, 1)).float(),
                         requires_grad=True).to(self.device)
        scaling_matrix = torch.cat([
            torch.cat([scale, zeros,  zeros, zeros], dim=2),
            torch.cat([zeros, scale,  zeros, zeros], dim=2),
            torch.cat([zeros, zeros,  scale, zeros], dim=2),
            torch.cat([zeros, zeros,  zeros, ones],  dim=2)], dim=1)

        # Translation matrix
        x_shift = Variable(torch.as_tensor(view_params[:,3].reshape(-1, 1, 1)).float(),
                           requires_grad=True).to(self.device)
        y_shift = Variable(torch.as_tensor(view_params[:,4].reshape(-1, 1, 1)).float(),
                           requires_grad=True).to(self.device)
        z_shift = Variable(torch.as_tensor(view_params[:,5].reshape(-1, 1, 1)).float(),
                           requires_grad=True).to(self.device)
        translation_matrix = torch.cat([
            torch.cat([ones,  zeros, zeros, x_shift], dim=2),
            torch.cat([zeros, ones,  zeros, y_shift], dim=2),
            torch.cat([zeros, zeros, ones,  z_shift], dim=2),
            torch.cat([zeros, zeros, zeros, ones],    dim=2)], dim=1)

        transformation_matrix = torch.matmul(translation_matrix, scaling_matrix)
        transformation_matrix = torch.matmul(transformation_matrix, rotation_matrix)

        return self.apply_transformation(voxel_array, transformation_matrix, size, new_size)

    def apply_transformation(self, voxel_array, transformation_matrix, size=16, new_size=16):

        batch_size = voxel_array.shape[0]
        # Aligning the centroid of the object (voxel grid) to origin for rotation,
        # then move the centroid back to the original position of the grid centroid
        centroid = Variable(torch.tensor([[1, 0, 0, -size * 0.5],
                                          [0, 1, 0, -size * 0.5],
                                          [0, 0, 1, -size * 0.5],
                                          [0, 0, 0,           1]]),
                                          requires_grad=True).to(self.device)
        centroid = centroid.reshape(1, 4, 4).repeat(batch_size, 1, 1)

        # However, since the rotated grid might be out of bound for the original grid size,
        # move the rotated grid to a new bigger grid
        centroid_new = Variable(torch.tensor([[1, 0, 0, new_size * 0.5],
                                              [0, 1, 0, new_size * 0.5],
                                              [0, 0, 1, new_size * 0.5],
                                              [0, 0, 0,              1]]),
                                              requires_grad=True).to(self.device)
        centroid_new = centroid_new.reshape(1, 4, 4).repeat(batch_size, 1, 1)

        transformed_centoid = torch.matmul(centroid_new, transformation_matrix)
        transformed_centoid = torch.matmul(transformed_centoid, centroid)
        transformed_centoid = transformed_centoid.inverse()
        #Ignore the homogenous coordinate so the results are 3D vectors
        #transformed_centoid = transformed_centoid[:, 0:3, :]

        grid = self.meshgrid(new_size, new_size, new_size)
        grid = grid.reshape(1, grid.shape[0], grid.shape[1])
        grid = grid.repeat(batch_size, 1, 1)

        grid_transform = torch.matmul(transformed_centoid, grid)
        x_flat = grid_transform[:, 0, :].reshape(-1)
        y_flat = grid_transform[:, 1, :].reshape(-1)
        z_flat = grid_transform[:, 2, :].reshape(-1)

        n_channels = voxel_array.shape[1]
        """
        out_shape = (batch_size, n_channels, new_size, new_size, new_size)
        transformed = self.interpolation(voxel_array, x_flat, y_flat, z_flat, new_size)
        transformed = transformed.reshape(out_shape)
        """
        transformed = self.interpolation(voxel_array, x_flat, y_flat, z_flat, new_size)
        out_shape = (batch_size, new_size, new_size, new_size, n_channels)
        transformed = transformed.reshape(out_shape).permute(0, 4, 1, 2, 3)
        return transformed

    def interpolation(self, voxel_array, x, y, z, size):
        batch_size, n_channels, height, width, depth = voxel_array.shape

        # do sampling
        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1
        z0 = torch.floor(z).long()
        z1 = z0 + 1

        x0 = torch.clamp(x0, 0, width-1)
        x1 = torch.clamp(x1, 0, width-1)
        y0 = torch.clamp(y0, 0, height-1)
        y1 = torch.clamp(y1, 0, height-1)
        z0 = torch.clamp(z0, 0, depth-1)
        z1 = torch.clamp(z1, 0, depth-1)

        rep  = torch.ones(1, size * size * size).long()
        base = torch.arange(batch_size) * width * height * depth
        base = torch.matmul(base.reshape(-1, 1), rep).reshape(-1).to(self.device)

        #Find the Z element of each index
        base_z0 = base + z0 * width * height
        base_z1 = base + z1 * width * height

        #Find the Y element based on Z
        base_z0_y0 = base_z0 + y0 * width
        base_z0_y1 = base_z0 + y1 * width
        base_z1_y0 = base_z1 + y0 * width
        base_z1_y1 = base_z1 + y1 * width

        # Find the X element based on Y, Z for Z=0
        idx_a = (base_z0_y0 + x0)
        idx_b = (base_z0_y1 + x0)
        idx_c = (base_z0_y0 + x1)
        idx_d = (base_z0_y1 + x1)

        # Find the X element based on Y,Z for Z =1
        idx_e = (base_z1_y0 + x0)
        idx_f = (base_z1_y1 + x0)
        idx_g = (base_z1_y0 + x1)
        idx_h = (base_z1_y1 + x1)

        # use indices to lookup pixels in the flat image and restore channels dim
        voxel_flat = voxel_array.permute(0, 2, 3, 4, 1).reshape(-1, n_channels)
        #voxel_flat = voxel_array.reshape(-1, n_channels)
        Ia = voxel_flat[idx_a]
        Ib = voxel_flat[idx_b]
        Ic = voxel_flat[idx_c]
        Id = voxel_flat[idx_d]
        Ie = voxel_flat[idx_e]
        If = voxel_flat[idx_f]
        Ig = voxel_flat[idx_g]
        Ih = voxel_flat[idx_h]

        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()
        z0_f = z0.float()
        z1_f = z1.float()

        #First slice XY along Z where z=0
        wa = ((x1_f - x) * (y1_f - y) * (z1_f - z)).unsqueeze(1)
        wb = ((x1_f - x) * (y - y0_f) * (z1_f - z)).unsqueeze(1)
        wc = ((x - x0_f) * (y1_f - y) * (z1_f - z)).unsqueeze(1)
        wd = ((x - x0_f) * (y - y0_f) * (z1_f - z)).unsqueeze(1)

        # First slice XY along Z where z=1
        we = ((x1_f - x) * (y1_f - y) * (z - z0_f)).unsqueeze(1)
        wf = ((x1_f - x) * (y - y0_f) * (z - z0_f)).unsqueeze(1)
        wg = ((x - x0_f) * (y1_f - y) * (z - z0_f)).unsqueeze(1)
        wh = ((x - x0_f) * (y - y0_f) * (z - z0_f)).unsqueeze(1)

        target = wa * Ia + wb * Ib + wc * Ic + wd * Id + we * Ie + wf * If + wg * Ig + wh * Ih
        return target

    def meshgrid(self, height, width, depth):
        z, y, x = torch.meshgrid(torch.arange(depth).to(self.device),
                                 torch.arange(height).to(self.device),
                                 torch.arange(width).to(self.device))
        x_flat = x.reshape(1, -1).float()
        y_flat = y.reshape(1, -1).float()
        z_flat = z.reshape(1, -1).float()
        ones = torch.ones(x_flat.shape).float().to(self.device)
        return torch.cat([x_flat, y_flat, z_flat, ones], dim=0)

def AdaIn(features, scale, bias):
    shape = features.shape
    new_shape = tuple(list(shape)[:2] + [1] * (len(shape)-2))
    moments = features.view(shape[0], shape[1], -1)
    mean = moments.mean(2).reshape(new_shape)
    variance = moments.var(2).reshape(new_shape)
    sigma = torch.rsqrt(variance + 1e-8)
    normalized = (features - mean) * sigma
    scale_broadcast = scale.reshape(mean.shape)
    bias_broadcast = bias.reshape(mean.shape)
    normalized = scale_broadcast * normalized
    normalized += bias_broadcast
    return normalized
