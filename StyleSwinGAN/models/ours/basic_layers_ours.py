import torch
import torch.nn as nn
import numpy as np


class ConstantInput(nn.Module):
    """
    Generator takes constant [B x H x W x C] dimensional feature map and gradually
    upsamples this feature map through cascade of transformer blocks.

    Important Note: Every batch will get the same constant input.

    Args:
        dim: Spatial dimension, i.e dimension of H and W.
        channel_dim: dimension across channels.

    Returns:
        constant_input: Initial feature map of size [B x H x W x C]

    """

    def __init__(self, dim, channel_dim):
        super().__init__()
        self.dim = dim
        self.channel_dim = channel_dim
        # Gaussian initialization, but self.const is learnt across the dataset =>
        # what kind of template to start with the generation process
        self.const = nn.Parameter(torch.randn(1, dim, dim, channel_dim))

    def forward(self, batch_size):
        constant_input = self.const.repeat(
            batch_size, 1, 1, 1
        )  # starting a template to generate img
        return constant_input


class AdaIN(nn.Module):
    """
    Input: [B x C x H x W]
    """

    def __init__(self, in_channels, style_dim):
        super().__init__()
        self.proj = nn.Linear(style_dim, 2 * in_channels)
        self.norm = nn.InstanceNorm1d(in_channels)

    def forward(self, input, style):
        style = self.proj(style).unsqueeze(-1)
        mean, std = style.chunk(2, 1)  # [B x C x 1] - [B x C x 1]

        B, C, _, _ = input.size()
        normalized = self.norm(input.view(B, C, -1))
        normalized = std * normalized + mean
        return normalized.transpose(-1, -2)


class Upsample(nn.Module):
    """
    Upsamples resolution with scale_factor 2
    eg. [4 x 4] => [8 x 8]
    torch.nn.Upsample upsamples last 2 dimensions as a result we have to permute the dimensions

    c_in: channel dim input
    c_out: channel dim output

    after upsampling H x W => channel dim reduced from c_in => c_out
    also layer normalization applied
    """

    def __init__(self, permute):
        super().__init__()
        self.permute = permute
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        """
        Expected Size: [B x H x W x C]
        Output Size: [B x 2H x 2W x C]
        """
        if self.permute:
            x = x.permute(0, 3, 1, 2)  # [B x C x H x W]
            x = self.upsample(x)  # [B x C x 2H x 2W]
            x = x.permute(0, 2, 3, 1)  # [B x 2H x 2W x C]
        else:
            x = self.upsample(x)  # [B x C x 2H x 2W]

        return x


class tRGB(nn.Module):
    """
    Expected Size: [B x H x W x C]
    Returns: [B x H x W x 3]
    """

    def __init__(self, in_channels, kernel_size=1):
        super().__init__()
        self.downsample = nn.Conv2d(in_channels, 3, kernel_size=kernel_size)

    def forward(self, x):
        B, L, C = x.size()
        D = int(L ** 0.5)
        return self.downsample(x.view(B, D, D, C).permute(0, 3, 1, 2))


class PixelNorm(nn.Module):
    """
    Expects [batch_size x style_dim] tensor
    Outputs normalized style vectors within a batch sample
    """

    def __init__(self):
        super().__init__()

    def forward(self, input):
        eps = 1e-8
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + eps)


class StyleMassage(nn.Module):
    def __init__(self, n_layers, style_dim, activation=nn.ReLU):
        super().__init__()
        mlp_list = [PixelNorm()]

        for _ in range(n_layers):
            mlp_list.append(nn.Linear(style_dim, style_dim))
            mlp_list.append(activation())

        self.net = nn.Sequential(*mlp_list)

    def forward(self, style):
        return self.net(style)


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qk_scale=None, attn_drop=0.0):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        torch.nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.attn_drop = nn.Dropout(attn_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: queries with shape of (num_windows*B, N, C)
            k: keys with shape of (num_windows*B, N, C)
            v: values with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = q.shape
        q = q.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.hidden_features = hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class StyleSwinTransformerBlock(nn.Module):
    r"""StyleSwin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        style_dim (int): Dimension of style vector.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        act_layer=nn.GELU,
        style_dim=512,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.shift_size = self.window_size // 2
        self.style_dim = style_dim
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        self.norm1 = AdaIN(dim, style_dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn = nn.ModuleList(
            [
                WindowAttention(
                    dim // 2,
                    window_size=(self.window_size, self.window_size),
                    num_heads=num_heads // 2,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                ),
                WindowAttention(
                    dim // 2,
                    window_size=(self.window_size, self.window_size),
                    num_heads=num_heads // 2,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                ),
            ]
        )

        attn_mask1 = None
        attn_mask2 = None
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask2 = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask2 = attn_mask2.masked_fill(
                attn_mask2 != 0, float(-100.0)
            ).masked_fill(attn_mask2 == 0, float(0.0))

        self.register_buffer("attn_mask1", attn_mask1)
        self.register_buffer("attn_mask2", attn_mask2)

        self.norm2 = AdaIN(dim, style_dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, style):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # Double Attn
        shortcut = x
        x = self.norm1(x.transpose(-1, -2).view(B, C, H, W), style).transpose(-1, -2)
        qkv = self.qkv(x.permute(0, 2, 1))
        qkv = qkv.reshape(B, -1, 3, C).permute(2, 0, 1, 3).reshape(3 * B, H, W, C)
        qkv_1 = qkv[:, :, :, : C // 2].reshape(3, B, H, W, C // 2)
        if self.shift_size > 0:
            qkv_2 = torch.roll(
                qkv[:, :, :, C // 2 :],
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            ).reshape(3, B, H, W, C // 2)
        else:
            qkv_2 = qkv[:, :, :, C // 2 :].reshape(3, B, H, W, C // 2)

        q1_windows, k1_windows, v1_windows = self.get_window_qkv(qkv_1)
        q2_windows, k2_windows, v2_windows = self.get_window_qkv(qkv_2)

        x1 = self.attn[0](q1_windows, k1_windows, v1_windows, self.attn_mask1)
        x2 = self.attn[1](q2_windows, k2_windows, v2_windows, self.attn_mask2)

        x1 = window_reverse(
            x1.view(-1, self.window_size * self.window_size, C // 2),
            self.window_size,
            H,
            W,
        )
        x2 = window_reverse(
            x2.view(-1, self.window_size * self.window_size, C // 2),
            self.window_size,
            H,
            W,
        )

        if self.shift_size > 0:
            x2 = torch.roll(x2, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x2 = x2

        x = torch.cat(
            [x1.reshape(B, H * W, C // 2), x2.reshape(B, H * W, C // 2)], dim=2
        )
        x = self.proj(x)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x.transpose(-1, -2).view(B, C, H, W), style))
        return x

    def get_window_qkv(self, qkv):
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, H, W, C
        C = q.shape[-1]
        q_windows = window_partition(q, self.window_size).view(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C
        k_windows = window_partition(k, self.window_size).view(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C
        v_windows = window_partition(v, self.window_size).view(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C
        return q_windows, k_windows, v_windows

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self,
                 embedding_dim,
                 padding_idx,
                 init_size=1024,
                 div_half_dim=False,
                 center_shift=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.div_half_dim = div_half_dim
        self.center_shift = center_shift

        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size, embedding_dim, padding_idx, self.div_half_dim)

        self.register_buffer('_float_tensor', torch.FloatTensor(1))

        self.max_positions = int(1e5)

    @staticmethod
    def get_embedding(num_embeddings,
                      embedding_dim,
                      padding_idx=None,
                      div_half_dim=False):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        assert embedding_dim % 2 == 0, (
            'In this version, we request '
            f'embedding_dim divisible by 2 but got {embedding_dim}')

        # there is a little difference from the original paper.
        half_dim = embedding_dim // 2
        if not div_half_dim:
            emb = np.log(10000) / (half_dim - 1)
        else:
            emb = np.log(1e4) / half_dim
        # compute exp(-log10000 / d * i)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(
            num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)],
                        dim=1).view(num_embeddings, -1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        return emb

    def forward(self, input, **kwargs):
        """Input is expected to be of size [bsz x seqlen].
        Returned tensor is expected to be of size  [bsz x seq_len x emb_dim]
        """
        assert input.dim() == 2 or input.dim(
        ) == 4, 'Input dimension should be 2 (1D) or 4(2D)'

        if input.dim() == 4:
            return self.make_grid2d_like(input, **kwargs)

        b, seq_len = input.shape
        max_pos = self.padding_idx + 1 + seq_len

        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embedding if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx)
        self.weights = self.weights.to(self._float_tensor)

        positions = self.make_positions(input, self.padding_idx).to(
            self._float_tensor.device)

        return self.weights.index_select(0, positions.view(-1)).view(
            b, seq_len, self.embedding_dim).detach()

    def make_positions(self, input, padding_idx):
        mask = input.ne(padding_idx).int()
        return (torch.cumsum(mask, dim=1).type_as(mask) *
                mask).long() + padding_idx

    def make_grid2d(self, height, width, num_batches=1, center_shift=None):
        h, w = height, width
        # if `center_shift` is not given from the outside, use
        # `self.center_shift`
        if center_shift is None:
            center_shift = self.center_shift

        h_shift = 0
        w_shift = 0
        # center shift to the input grid
        if center_shift is not None:
            # if h/w is even, the left center should be aligned with
            # center shift
            if h % 2 == 0:
                h_left_center = h // 2
                h_shift = center_shift - h_left_center
            else:
                h_center = h // 2 + 1
                h_shift = center_shift - h_center

            if w % 2 == 0:
                w_left_center = w // 2
                w_shift = center_shift - w_left_center
            else:
                w_center = w // 2 + 1
                w_shift = center_shift - w_center

        # Note that the index is started from 1 since zero will be padding idx.
        # axis -- (b, h or w)
        x_axis = torch.arange(1, w + 1).unsqueeze(0).repeat(num_batches,
                                                            1) + w_shift
        y_axis = torch.arange(1, h + 1).unsqueeze(0).repeat(num_batches,
                                                            1) + h_shift

        # emb -- (b, emb_dim, h or w)
        x_emb = self(x_axis).transpose(1, 2)
        y_emb = self(y_axis).transpose(1, 2)

        # make grid for x/y axis
        # Note that repeat will copy data. If use learned emb, expand may be
        # better.
        x_grid = x_emb.unsqueeze(2).repeat(1, 1, h, 1)
        y_grid = y_emb.unsqueeze(3).repeat(1, 1, 1, w)

        # cat grid -- (b, 2 x emb_dim, h, w)
        grid = torch.cat([x_grid, y_grid], dim=1)
        return grid.detach()

    def make_grid2d_like(self, x, center_shift=None):
        """Input tensor with shape of (b, ..., h, w) Return tensor with shape
        of (b, 2 x emb_dim, h, w)
        Note that the positional embedding highly depends on the the function,
        ``make_positions``.
        """
        h, w = x.shape[-2:]
        grid = self.make_grid2d(h, w, x.size(0), center_shift)

        return grid.to(x)
