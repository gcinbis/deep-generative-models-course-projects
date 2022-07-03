import torch
from torch import nn
import numpy as np


class ITRANS(nn.Module):
    def __init__(self, num_channels):
        super(ITRANS, self).__init__()
        self.c = num_channels
        self.convq1 = torch.nn.Parameter(torch.randn([num_channels, num_channels, 1, 1]), requires_grad=True)
        self.s = torch.nn.Softmax(dim=-1)
        self.convk1 = torch.nn.Parameter(torch.randn([num_channels, num_channels, 1, 1]), requires_grad=True)
        self.register_parameter("offset", nn.Parameter(torch.ones([1, 1, 1]) * 1.01))
        self.register_parameter("scale", nn.Parameter(torch.ones([1, 1, 1]) * 10))
    def forward(self, input: torch.Tensor, logdet = 0, reverse = False, permute = False):
        # checkerboard
        B, C, H, W = input.shape

        self.mask = torch.tensor(np.ones((B, C, H * W), dtype=np.float64)).cuda()

        ones = torch.tensor(np.ones((1, 1, H * W), dtype=np.float64))
        ones.flatten()[::2] = 0
        zeros = torch.tensor(np.zeros((1, 1, H * W), dtype=np.float64))
        zeros.flatten()[::2] = 1
        grid = torch.cat((ones, zeros))
        cat_ones = torch.cat((ones, zeros))

        for i in range(0, C // 2 - 1):
            grid = torch.cat((grid, cat_ones))
        grid = grid.view(1, C, H * W)

        checkerboard = torch.cat((grid, grid))
        checkerboard_1 = torch.cat((grid, grid))

        for i in range(1, B // 2):
            checkerboard = torch.cat((checkerboard, checkerboard_1))

        if B == 1:
            checkerboard = grid

        self.mask = checkerboard.type(torch.FloatTensor).to(input.device)

        mask = self.mask
        if not reverse:
            p = input.shape[-1] // 2

            input_mask = input.view(B, C, H*W).type(torch.FloatTensor).to(input.device) * mask.to(input.device)
            reverse_rearranged_input_mask = input_mask.view(B, C, H, W).type(torch.FloatTensor).to(input.device)

            q1 = torch.nn.functional.conv2d(reverse_rearranged_input_mask, self.convq1.type(torch.FloatTensor).to(input.device))
            k1 = torch.nn.functional.conv2d(reverse_rearranged_input_mask, self.convk1.type(torch.FloatTensor).to(input.device))

            full_inp_q1 = q1.view(B, C, H*W).to(input.device)
            full_inp_k1 = k1.view(B, C, H*W).to(input.device)

            attn = self.s((torch.matmul(full_inp_q1, full_inp_k1.permute(0,2,1))/self.scale.to(input.device))).to(input.device)

            id = torch.eye(attn.shape[-1]).to(input.device) * self.offset.to(input.device)
            logdet_trans = torch.slogdet(attn + id)[1] * p * (p//2) * self.c
            logdet = logdet + logdet_trans
            out_attn = torch.matmul(attn, input.view(B, C, H*W) * mask)
            out = out_attn*(1-mask) + input_mask
            output = out.view(B, C, H, W)
        else:
            p = input.shape[-1] // 2
            out = input.view(B, C, H*W)
            if permute:
                mask = 1 - mask
            rev = out * mask
            rev_rearrange = rev.view(B, C, H, W)
            q1 = torch.nn.functional.conv2d(rev_rearrange.type(torch.FloatTensor).to(input.device), self.convq1.to(input.device))
            k1 = torch.nn.functional.conv2d(rev_rearrange.type(torch.FloatTensor).to(input.device), self.convk1.to(input.device))
            full_inp_q1_rev = q1.view(B, C, H*W)
            full_inp_k1_rev = k1.view(B, C, H*W)
            attn_mask = mask
            attn = (self.s((torch.matmul(full_inp_q1_rev,full_inp_k1_rev.permute(0,2,1))/self.scale.to(input.device))))

            id = torch.eye(attn.shape[-1]).to(input.device) * self.offset.to(input.device)
            logdet_trans = torch.slogdet(attn + id)[1] * p * (p//2) * self.c
            logdet = logdet - logdet_trans
            attn_inv = torch.inverse(attn + id)
            out_attn = torch.matmul(attn_inv.type(torch.FloatTensor).to(input.device), out.type(torch.FloatTensor).to(input.device)*(1-mask).type(torch.FloatTensor).to(input.device))
            output = out_attn*(1-mask) + out*mask
            output = output.view(B,C,H,W)

        return output, logdet