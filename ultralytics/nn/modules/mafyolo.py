# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

__all__ = ("RepHDW",)


class Conv(nn.Module):
    """Normal Conv with SiLU activation."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, groups=1, bias=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class AVG(nn.Module):
    def __init__(self, down_n=2):
        super().__init__()
        torch.use_deterministic_algorithms(True, warn_only=True)
        self.avg_pool = nn.functional.adaptive_avg_pool2d
        self.down_n = down_n
        # self.output_size = np.array([H, W])

    def forward(self, x):
        _B, _C, H, W = x.shape
        H = int(H / self.down_n)
        W = int(W / self.down_n)
        output_size = np.array([H, W])
        x = self.avg_pool(x, output_size)
        return x


class Dual_in(nn.Module):
    def __init__(self, c2=6):
        super().__init__()

    def forward(self, x):
        return x


class Dual_out(nn.Module):
    def __init__(self, idx):
        self.idx = idx
        super().__init__()

    def forward(self, x):
        x1, x2 = x[:, :3, :, :], x[:, 3:, :, :]
        if self.idx == 1:
            x = x1
        else:
            x = x2
        return x


class Add(nn.Module):
    def __init__(self, c2):
        super().__init__()

    def forward(self, x):
        return x[0] + x[1]


class RepHDW(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        depth=1,
        shortcut=True,
        expansion=0.5,
        kersize=5,
        depth_expansion=1,
        small_kersize=3,
        use_depthwise=True,
    ):
        super().__init__()
        c1 = int(out_channels * expansion) * 2
        c_ = int(out_channels * expansion)
        self.c_ = c_
        self.conv1 = Conv(in_channels, c1, 1, 1)
        self.m = nn.ModuleList(
            DepthBottleneckUni(self.c_, self.c_, shortcut, kersize, depth_expansion, small_kersize, use_depthwise)
            for _ in range(depth)
        )
        self.conv2 = Conv(c_ * (depth + 2), out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x_out = list(x.split((self.c_, self.c_), 1))
        for conv in self.m:
            y = conv(x_out[-1])
            x_out.append(y)
        y_out = torch.cat(x_out, axis=1)
        y_out = self.conv2(y_out)
        return y_out


class DepthBottleneckUni(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        kersize=5,
        expansion_depth=1,
        small_kersize=3,
        use_depthwise=True,
    ):
        super().__init__()

        mid_channel = int(in_channels * expansion_depth)
        self.conv1 = Conv(in_channels, mid_channel, 1)
        self.shortcut = shortcut
        if use_depthwise:
            self.conv2 = UniRepLKNetBlock(mid_channel, kernel_size=kersize)
            self.act = nn.SiLU()
            self.one_conv = Conv(mid_channel, out_channels, kernel_size=1)

        else:
            self.conv2 = Conv(out_channels, out_channels, 3, 1)

    def forward(self, x):
        y = self.conv1(x)

        y = self.act(self.conv2(y))

        y = self.one_conv(y)
        return y


class UniRepLKNetBlock(nn.Module):
    def __init__(self, dim, kernel_size, deploy=False, attempt_use_lk_impl=True):
        super().__init__()
        if deploy:
            print("------------------------------- Note: deploy mode")
        if kernel_size == 0:
            self.dwconv = nn.Identity()
        elif kernel_size >= 3:
            self.dwconv = DilatedReparamBlock(dim, kernel_size, deploy=deploy, attempt_use_lk_impl=attempt_use_lk_impl)
        else:
            assert kernel_size in [3]
            self.dwconv = get_conv2d_uni(
                dim,
                dim,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                dilation=1,
                groups=dim,
                bias=deploy,
                attempt_use_lk_impl=attempt_use_lk_impl,
            )

        if deploy or kernel_size == 0:
            self.norm = nn.Identity()
        else:
            self.norm = get_bn(dim)

    def forward(self, inputs):

        out = self.norm(self.dwconv(inputs))
        return out

    def reparameterize(self):
        if hasattr(self.dwconv, "merge_dilated_branches"):
            self.dwconv.merge_dilated_branches()
        if hasattr(self.norm, "running_var"):
            std = (self.norm.running_var + self.norm.eps).sqrt()
            if hasattr(self.dwconv, "lk_origin"):
                self.dwconv.lk_origin.weight.data *= (self.norm.weight / std).view(-1, 1, 1, 1)
                self.dwconv.lk_origin.bias.data = (
                    self.norm.bias + (self.dwconv.lk_origin.bias - self.norm.running_mean) * self.norm.weight / std
                )
            else:
                conv = nn.Conv2d(
                    self.dwconv.in_channels,
                    self.dwconv.out_channels,
                    self.dwconv.kernel_size,
                    self.dwconv.padding,
                    self.dwconv.groups,
                    bias=True,
                )
                conv.weight.data = self.dwconv.weight * (self.norm.weight / std).view(-1, 1, 1, 1)
                conv.bias.data = self.norm.bias - self.norm.running_mean * self.norm.weight / std
                self.dwconv = conv
            self.norm = nn.Identity()


class DilatedReparamBlock(nn.Module):
    """Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet) We assume the inputs to
    this block are (N, C, H, W).
    """

    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d_uni(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            dilation=1,
            groups=channels,
            bias=deploy,
        )
        self.attempt_use_lk_impl = attempt_use_lk_impl

        if kernel_size == 17:
            self.kernel_sizes = [5, 9, 3, 3, 3]
            self.dilates = [1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [7, 5, 3]
            self.dilates = [1, 1, 1]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3]
            self.dilates = [1, 1]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 1]
            self.dilates = [1, 1]
        elif kernel_size == 3:
            self.kernel_sizes = [3, 1]
            self.dilates = [1, 1]

        else:
            raise ValueError("Dilated Reparam Block requires kernel_size >= 5")

        if not deploy:
            self.origin_bn = get_bn(channels)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__(
                    f"dil_conv_k{k}_{r}",
                    nn.Conv2d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=k,
                        stride=1,
                        padding=(r * (k - 1) + 1) // 2,
                        dilation=r,
                        groups=channels,
                        bias=False,
                    ),
                )
                self.__setattr__(f"dil_bn_k{k}_{r}", get_bn(channels))

    def forward(self, x):
        if not hasattr(self, "origin_bn"):  # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__(f"dil_conv_k{k}_{r}")
            bn = self.__getattr__(f"dil_bn_k{k}_{r}")
            out = out + bn(conv(x))
        return out

    def merge_dilated_branches(self):
        if hasattr(self, "origin_bn"):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__(f"dil_conv_k{k}_{r}")
                bn = self.__getattr__(f"dil_bn_k{k}_{r}")
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d_uni(
                origin_k.size(0),
                origin_k.size(0),
                origin_k.size(2),
                stride=1,
                padding=origin_k.size(2) // 2,
                dilation=1,
                groups=origin_k.size(0),
                bias=True,
                attempt_use_lk_impl=self.attempt_use_lk_impl,
            )
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__("origin_bn")
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__(f"dil_conv_k{k}_{r}")
                self.__delattr__(f"dil_bn_k{k}_{r}")


import collections.abc
from itertools import repeat


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


def get_conv2d_uni(
    in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, attempt_use_lk_impl=True
):
    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)
    kernel_size[0] == kernel_size[1] and kernel_size[0] > 5 and padding == (kernel_size[0] // 2, kernel_size[1] // 2)

    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )


def convert_dilated_to_nondilated(kernel, dilate_rate):
    identity_kernel = torch.ones((1, 1, 1, 1), dtype=kernel.dtype, device=kernel.device)
    if kernel.size(1) == 1:
        #   This is a DW kernel
        dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
        return dilated
    else:
        #   This is a dense or group-wise (but not DW) kernel
        slices = []
        for i in range(kernel.size(1)):
            dilated = F.conv_transpose2d(kernel[:, i : i + 1, :, :], identity_kernel, stride=dilate_rate)
            slices.append(dilated)
        return torch.cat(slices, dim=1)


def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    return merged_kernel


def get_bn(channels):
    return nn.BatchNorm2d(channels)


class DepthBottleneckUniv2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        kersize=5,
        expansion_depth=1,
        small_kersize=3,
        use_depthwise=True,
    ):
        super().__init__()

        mid_channel = int(in_channels * expansion_depth)
        mid_channel2 = mid_channel
        self.conv1 = Conv(in_channels, mid_channel, 1)
        self.shortcut = shortcut
        if use_depthwise:
            self.conv2 = UniRepLKNetBlock(mid_channel, kernel_size=kersize)
            self.act = nn.SiLU()
            self.one_conv = Conv(mid_channel, mid_channel2, kernel_size=1)

            self.conv3 = UniRepLKNetBlock(mid_channel2, kernel_size=kersize)
            self.act1 = nn.SiLU()
            self.one_conv2 = Conv(mid_channel2, out_channels, kernel_size=1)
        else:
            self.conv2 = Conv(out_channels, out_channels, 3, 1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.act(self.conv2(y))
        y = self.one_conv(y)
        y = self.act1(self.conv3(y))
        y = self.one_conv2(y)
        return y


class RepHMS(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        width=3,
        depth=1,
        depth_expansion=2,
        kersize=5,
        shortcut=True,
        expansion=0.5,
        small_kersize=3,
        use_depthwise=True,
    ):
        super().__init__()
        self.width = width
        self.depth = depth
        c1 = int(out_channels * expansion) * width
        c_ = int(out_channels * expansion)
        self.c_ = c_
        self.conv1 = Conv(in_channels, c1, 1, 1)
        self.RepElanMSBlock = nn.ModuleList()
        for _ in range(width - 1):
            DepthBlock = nn.ModuleList(
                [
                    DepthBottleneckUniv2(
                        self.c_, self.c_, shortcut, kersize, depth_expansion, small_kersize, use_depthwise
                    )
                    for _ in range(depth)
                ]
            )
            self.RepElanMSBlock.append(DepthBlock)

        self.conv2 = Conv(c_ * 1 + c_ * (width - 1) * depth, out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x_out = [x[:, i * self.c_ : (i + 1) * self.c_] for i in range(self.width)]
        x_out[1] = x_out[1] + x_out[0]
        cascade = []
        elan = [x_out[0]]
        for i in range(self.width - 1):
            for j in range(self.depth):
                if i > 0:
                    x_out[i + 1] = x_out[i + 1] + cascade[j]
                    if j == self.depth - 1:
                        # cascade = [cascade[-1]]
                        if self.depth > 1:
                            cascade = [cascade[-1]]
                        else:
                            cascade = []
                x_out[i + 1] = self.RepElanMSBlock[i][j](x_out[i + 1])
                elan.append(x_out[i + 1])
                if i < self.width - 2:
                    cascade.append(x_out[i + 1])

        y_out = torch.cat(elan, 1)
        y_out = self.conv2(y_out)
        return y_out


class DepthBottleneckv2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        kersize=5,
        expansion_depth=1,
        small_kersize=3,
        use_depthwise=True,
    ):
        super().__init__()

        mid_channel = int(in_channels * expansion_depth)
        mid_channel2 = mid_channel
        self.conv1 = Conv(in_channels, mid_channel, 1)
        self.shortcut = shortcut
        if use_depthwise:
            self.conv2 = DWConv(mid_channel, mid_channel, kersize)
            # self.act = nn.SiLU()
            self.one_conv = Conv(mid_channel, mid_channel2, kernel_size=1)

            self.conv3 = DWConv(mid_channel2, mid_channel2, kersize)
            # self.act1 = nn.SiLU()
            self.one_conv2 = Conv(mid_channel2, out_channels, kernel_size=1)
        else:
            self.conv2 = Conv(out_channels, out_channels, 3, 1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.one_conv(y)
        y = self.conv3(y)
        y = self.one_conv2(y)
        return y


class ConvMS(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        width=3,
        depth=1,
        depth_expansion=2,
        kersize=5,
        shortcut=True,
        expansion=0.5,
        small_kersize=3,
        use_depthwise=True,
    ):
        super().__init__()
        self.width = width
        self.depth = depth
        c1 = int(out_channels * expansion) * width
        c_ = int(out_channels * expansion)
        self.c_ = c_
        self.conv1 = Conv(in_channels, c1, 1, 1)
        self.RepElanMSBlock = nn.ModuleList()
        for _ in range(width - 1):
            DepthBlock = nn.ModuleList(
                [
                    DepthBottleneckv2(
                        self.c_, self.c_, shortcut, kersize, depth_expansion, small_kersize, use_depthwise
                    )
                    for _ in range(depth)
                ]
            )
            self.RepElanMSBlock.append(DepthBlock)

        self.conv2 = Conv(c_ * 1 + c_ * (width - 1) * depth, out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x_out = [x[:, i * self.c_ : (i + 1) * self.c_] for i in range(self.width)]
        x_out[1] = x_out[1] + x_out[0]
        cascade = []
        elan = [x_out[0]]
        for i in range(self.width - 1):
            for j in range(self.depth):
                if i > 0:
                    x_out[i + 1] = x_out[i + 1] + cascade[j]
                    if j == self.depth - 1:
                        # cascade = [cascade[-1]]
                        if self.depth > 1:
                            cascade = [cascade[-1]]
                        else:
                            cascade = []
                x_out[i + 1] = self.RepElanMSBlock[i][j](x_out[i + 1])
                elan.append(x_out[i + 1])
                if i < self.width - 2:
                    cascade.append(x_out[i + 1])

        y_out = torch.cat(elan, 1)
        y_out = self.conv2(y_out)
        return y_out


class TFCF(nn.Module):
    def __init__(
        self, d_model, vert_anchors=16, horz_anchors=16, h=8, block_exp=4, n_layer=1, attn_pdrop=0.1, resid_pdrop=0.1
    ):
        super().__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        d_k = d_model
        d_v = d_model

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb_N = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
        self.pos_emb_A = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))

        self.avgpool_N = AdaptivePool2d(self.vert_anchors, self.horz_anchors, "avg")
        self.maxpool_N = AdaptivePool2d(self.vert_anchors, self.horz_anchors, "max")

        self.avgpool_A = AdaptivePool2d(self.vert_anchors, self.horz_anchors, "avg")
        self.maxpool_A = AdaptivePool2d(self.vert_anchors, self.horz_anchors, "max")

        # LearnableCoefficient
        self.learnable_N = LearnableWeights()
        self.learnable_A = LearnableWeights()

        # self.gtfa = GTFA(d_model)
        # init weights
        self.apply(self._init_weights)
        # cross transformer
        self.crosstransformer = nn.Sequential(
            *[CrossTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop) for i in range(n_layer)]
        )
        self.concat = Concat(dimension=1)
        self.conv1x1_out = Conv_trans(c1=d_model * 2, c2=d_model, k=1, s=1, p=0, g=1, act=True)
        # GTFA

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        N_fea = x[0]
        A_fea = x[1]
        # N_fea, A_fea = self.gtfa([N_fea, A_fea])

        assert N_fea.shape[0] == A_fea.shape[0]
        bs, _c, h, w = N_fea.shape

        new_N_fea = self.learnable_N(self.avgpool_N(N_fea), self.maxpool_N(N_fea))
        new_c, new_h, new_w = new_N_fea.shape[1], new_N_fea.shape[2], new_N_fea.shape[3]
        N_fea_flat = new_N_fea.flatten(2).transpose(1, 2) + self.pos_emb_N

        new_A_fea = self.learnable_A(self.avgpool_A(A_fea), self.maxpool_A(A_fea))
        A_fea_flat = new_A_fea.flatten(2).transpose(1, 2) + self.pos_emb_A
        N_fea_flat, A_fea_flat = self.crosstransformer([N_fea_flat, A_fea_flat])

        N_fea_CFE = N_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2).contiguous()
        if self.training:
            N_fea_CFE = F.interpolate(N_fea_CFE, size=([h, w]), mode="nearest")
        else:
            N_fea_CFE = F.interpolate(N_fea_CFE, size=([h, w]), mode="bilinear")
        new_N_fea = N_fea_CFE + N_fea

        A_fea_CFE = A_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2).contiguous()
        if self.training:
            A_fea_CFE = F.interpolate(A_fea_CFE, size=([h, w]), mode="nearest")
        else:
            A_fea_CFE = F.interpolate(A_fea_CFE, size=([h, w]), mode="bilinear")
        new_A_fea = A_fea_CFE + A_fea

        # new_N_fea, new_A_fea = self.gtfa([new_N_fea, new_A_fea])
        new_fea = self.concat([new_N_fea, new_A_fea])
        new_fea = self.conv1x1_out(new_fea)
        return new_fea


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        # print(x.shape)
        return torch.cat(x, self.d)


# class AdaptivePool2d(nn.Module):
#     def __init__(self, output_h, output_w, pool_type="avg"):
#         super(AdaptivePool2d, self).__init__()
#
#         self.output_h = output_h
#         self.output_w = output_w
#         self.pool_type = pool_type
#
#     def forward(self, x):
#         bs, c, input_h, input_w = x.shape
#
#         if (input_h > self.output_h) or (input_w > self.output_w):
#             self.stride_h = input_h // self.output_h
#             self.stride_w = input_w // self.output_w
#             self.kernel_size = (
#                 input_h - (self.output_h - 1) * self.stride_h,
#                 input_w - (self.output_w - 1) * self.stride_w,
#             )
#
#             if self.pool_type == "avg":
#                 y = nn.AvgPool2d(kernel_size=self.kernel_size, stride=(self.stride_h, self.stride_w), padding=0)(x)
#             else:
#                 y = nn.MaxPool2d(kernel_size=self.kernel_size, stride=(self.stride_h, self.stride_w), padding=0)(x)
#         else:
#             y = x
#
#         return y
import torch.nn as nn


class AdaptivePool2d(nn.Module):
    def __init__(self, output_h, output_w, pool_type="avg"):
        super().__init__()
        self.output_h = output_h
        self.output_w = output_w
        self.pool_type = pool_type.lower()  # 统一转为小写
        assert pool_type in ["avg", "max"], "pool_type must be 'avg' or 'max'"

    def forward(self, x):
        _bs, _c, input_h, input_w = x.shape

        # 情况1：输入尺寸 ≤ 输出尺寸 → 直接返回或上采样
        if input_h <= self.output_h or input_w <= self.output_w:
            if input_h != self.output_h or input_w != self.output_w:
                # 自动选择上采样模式
                return nn.functional.interpolate(
                    x, size=(self.output_h, self.output_w), mode="bilinear", align_corners=False
                )
            return x

        # 情况2：输入尺寸 > 输出尺寸 → 动态计算池化参数
        self.stride_h = max(1, input_h // self.output_h)  # 步长至少为1
        self.stride_w = max(1, input_w // self.output_w)

        self.kernel_size = (
            input_h - (self.output_h - 1) * self.stride_h,
            input_w - (self.output_w - 1) * self.stride_w,
        )

        # 选择池化类型
        if self.pool_type == "avg":
            return nn.AvgPool2d(kernel_size=self.kernel_size, stride=(self.stride_h, self.stride_w), padding=0)(x)
        else:
            return nn.MaxPool2d(kernel_size=self.kernel_size, stride=(self.stride_h, self.stride_w), padding=0)(x)


import math


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv_trans(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class CrossTransformerBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, loops_num=1):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)
        """
        super().__init__()
        self.loops = loops_num
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.crossatt = CrossAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp_N = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            # nn.SiLU(),  # changed from GELU
            nn.GELU(),  # changed from GELU
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )
        self.ffn_vis = nn.Sequential(
            Conv_trans(d_model, d_model * 2, 1), Conv_trans(d_model * 2, d_model, 1, act=False)
        )
        self.mlp_A = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            # nn.SiLU(),  # changed from GELU
            nn.GELU(),  # changed from GELU
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )
        # Layer norm
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        # Learnable Coefficient
        self.coefficient1 = LearnableCoefficient()
        self.coefficient2 = LearnableCoefficient()
        self.coefficient3 = LearnableCoefficient()
        self.coefficient4 = LearnableCoefficient()
        self.coefficient5 = LearnableCoefficient()
        self.coefficient6 = LearnableCoefficient()
        self.coefficient7 = LearnableCoefficient()
        self.coefficient8 = LearnableCoefficient()

    def forward(self, x):
        N_fea_flat = x[0]
        A_fea_flat = x[1]
        assert N_fea_flat.shape[0] == A_fea_flat.shape[0]
        _bs, nx, _c = N_fea_flat.size()
        int(math.sqrt(nx))

        for loop in range(self.loops):
            # with Learnable Coefficient
            N_fea_out, A_fea_out = self.crossatt([N_fea_flat, A_fea_flat])
            N_att_out = self.coefficient1(N_fea_flat) + self.coefficient2(N_fea_out)
            A_att_out = self.coefficient3(A_fea_flat) + self.coefficient4(A_fea_out)
            N_fea_flat = self.coefficient5(N_att_out) + self.coefficient6(self.mlp_N(self.LN2(N_att_out)))
            A_fea_flat = self.coefficient7(A_att_out) + self.coefficient8(self.mlp_A(self.LN2(A_att_out)))

        return [N_fea_flat, A_fea_flat]


class LearnableCoefficient(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def forward(self, x):
        out = x * self.bias
        return out


class LearnableWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.w2 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

    def forward(self, x1, x2):
        out = x1 * self.w1 + x2 * self.w2
        return out


class CrossAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=0.1, resid_pdrop=0.1):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        """
        super().__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj_N = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj_N = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj_N = nn.Linear(d_model, h * self.d_v)  # value projection

        self.que_proj_A = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj_A = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj_A = nn.Linear(d_model, h * self.d_v)  # value projection

        self.out_proj_N = nn.Linear(h * self.d_v, d_model)  # output projection
        self.out_proj_A = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # layer norm
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        """Computes Self-Attention.

        Args:
            x (tensor): input (token) dim:(b_s, nx, c), b_s means batch size nx means length, for CNN, equals H*W, i.e.
                the length of feature maps c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).

        Returns:
            output (tensor): dim:(b_s, nx, c).
        """
        N_fea_flat = x[0]
        A_fea_flat = x[1]
        b_s, nq = N_fea_flat.shape[:2]
        nk = N_fea_flat.shape[1]

        # Self-Attention
        N_fea_flat = self.LN1(N_fea_flat)
        q_N = (
            self.que_proj_N(N_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        )  # (b_s, h, nq, d_k)
        k_N = (
            self.key_proj_N(N_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        )  # (b_s, h, d_k, nk) K^T
        v_N = (
            self.val_proj_N(N_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)
        )  # (b_s, h, nk, d_v)

        A_fea_flat = self.LN2(A_fea_flat)
        q_A = (
            self.que_proj_A(A_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        )  # (b_s, h, nq, d_k)
        k_A = (
            self.key_proj_A(A_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        )  # (b_s, h, d_k, nk) K^T
        v_A = (
            self.val_proj_A(A_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)
        )  # (b_s, h, nk, d_v)

        att_N = torch.matmul(q_A, k_N) / np.sqrt(self.d_k)
        att_A = torch.matmul(q_N, k_A) / np.sqrt(self.d_k)

        # get attention matrix
        att_N = torch.softmax(att_N, -1)
        att_N = self.attn_drop(att_N)
        att_A = torch.softmax(att_A, -1)
        att_A = self.attn_drop(att_A)

        # output
        out_N = (
            torch.matmul(att_N, v_N).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        )  # (b_s, nq, h*d_v)
        out_N = self.resid_drop(self.out_proj_N(out_N))  # (b_s, nq, d_model)
        out_A = (
            torch.matmul(att_A, v_A).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        )  # (b_s, nq, h*d_v)
        out_A = self.resid_drop(self.out_proj_A(out_A))  # (b_s, nq, d_model)

        return [out_N, out_A]


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style="lp", groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ["lp", "pl"]
        if style == "pl":
            assert in_channels >= scale**2 and in_channels % scale**2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == "pl":
            in_channels = in_channels // scale**2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale**2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            self.constant_init(self.scope, val=0.0)

        self.register_buffer("init_pos", self._init_pos())

    def normal_init(self, module, mean=0, std=1, bias=0):
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.normal_(module.weight, mean, std)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def constant_init(self, module, val, bias=0):
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.constant_(module.weight, val)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = (
            torch.stack(torch.meshgrid([coords_w, coords_h]))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
            .type(x.dtype)
            .to(x.device)
        )
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = (
            F.pixel_shuffle(coords.view(B, -1, H, W), self.scale)
            .view(B, 2, -1, self.scale * H, self.scale * W)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        return F.grid_sample(
            x.reshape(B * self.groups, -1, H, W), coords, mode="bilinear", align_corners=False, padding_mode="border"
        ).reshape((B, -1, self.scale * H, self.scale * W))

    def forward_lp(self, x):
        if hasattr(self, "scope"):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, "scope"):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == "pl":
            return self.forward_pl(x)
        return self.forward_lp(x)


class RepHEA(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        width=3,
        depth=1,
        depth_expansion=2,
        kersize=5,
        shortcut=True,
        expansion=0.5,
        small_kersize=3,
        use_depthwise=True,
    ):
        super().__init__()
        self.width = width
        self.depth = depth
        c1 = int(out_channels * expansion) * width
        c_ = int(out_channels * expansion)
        c__ = int(c_ * depth_expansion)
        self.c_ = c_
        self.conv1 = Conv(in_channels, c1, 1, 1)
        self.conv0_0 = Conv(c_, c__, 1)
        self.directional_blockH = UniRepSingleAxisConvBlock(
            dim=c__, kernel_sizes=[5, 7, 9, 11, 13], is_vertical=False, deploy=False
        )
        # self.Hsilu = nn.SiLU()
        self.directional_blockV = UniRepSingleAxisConvBlock(
            dim=c__, kernel_sizes=[5, 7, 9, 11, 13], is_vertical=True, deploy=False
        )
        # self.Vsilu = nn.SiLU()
        self.golbel = DepthBottleneckUniv3(
            self.c_, self.c_, shortcut, kersize, depth_expansion, small_kersize, use_depthwise
        )
        self.attention = MixedAttention(c__)
        self.project_out1 = Conv(c__, c_, 1)
        self.project_out2 = Conv(c__, c_, 1)
        self.conv2 = Conv(c_ * 1 + c_ * (width) * depth, out_channels, 1, 1)
        self.num_heads = 8

    def forward(self, x):
        _, _, _h, _w = x.shape
        x = self.conv1(x)
        x_out = [x[:, i * self.c_ : (i + 1) * self.c_] for i in range(self.width)]
        x_out1 = x_out[1]
        x_out1 = self.conv0_0(x_out1)
        y_out1 = self.directional_blockH(x_out1)
        y_out2 = self.directional_blockV(x_out1)
        y_out1 = self.project_out1(y_out1)
        x_out3 = x_out[2]
        x_out3 = self.golbel(x_out3)
        y_out2 = self.project_out2(y_out2)
        y_out = self.attention(y_out1, y_out2, x_out3)
        y_out = torch.cat([x_out[0], y_out], 1)
        y_out = self.conv2(y_out)
        return y_out


class MixedAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.weight_h = nn.Parameter(torch.ones(1))
        self.weight_v = nn.Parameter(torch.ones(1))
        self.weight_global = nn.Parameter(torch.ones(1))

    def forward(self, h_feat, v_feat, global_feat):
        # Normalize weights
        total_weight = self.weight_h + self.weight_v + self.weight_global
        w_h = self.weight_h / total_weight
        w_v = self.weight_v / total_weight
        w_g = self.weight_global / total_weight
        # print(w_g)

        return torch.cat([w_h * h_feat, w_v * v_feat, w_g * global_feat], 1)


class DepthBottleneckUniv3(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        kersize=5,
        expansion_depth=1,
        small_kersize=3,
        use_depthwise=True,
    ):
        super().__init__()

        mid_channel = int(in_channels * expansion_depth)
        mid_channel2 = mid_channel
        self.conv1 = Conv(in_channels, mid_channel, 1)
        self.shortcut = shortcut

        self.conv2 = UniRepLKNetBlock(mid_channel, kernel_size=kersize)
        # self.act = nn.SiLU()
        self.one_conv = Conv(mid_channel2, out_channels, kernel_size=1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        # y = self.act(self.conv2(y))
        y = self.one_conv(y)
        return y


class UniRepSingleAxisConvBlock(nn.Module):
    def __init__(self, dim, kernel_sizes, is_vertical=True, deploy=False, attempt_use_lk_impl=True):
        """与UniRepLKNetBlock类似，将多个单轴卷积并行，然后可在reparameterize时合并。 kernel_sizes例如：[5,7,9]表示并行三个卷积核。 is_vertical=True表示 1xK
        纵向卷积，否则为 Kx1 横向卷积。.
        """
        super().__init__()
        if deploy:
            print("------------------------------- Note: deploy mode")

        # 这里类似原始代码中对于dwconv的处理，直接用SingleAxisReparamBlock
        self.single_axis_block = SingleAxisReparamBlock(
            dim, kernel_sizes, is_vertical=is_vertical, deploy=deploy, attempt_use_lk_impl=attempt_use_lk_impl
        )

        # 如果deploy则不需要单独的norm
        # 注意：SingleAxisReparamBlock内部已处理BN与合并逻辑，这里不再需要单独的norm
        self.norm = nn.Identity()

    def forward(self, x):
        out = self.norm(self.single_axis_block(x))
        return out

    def reparameterize(self):
        # 合并单轴分支
        if hasattr(self.single_axis_block, "merge_single_axis_branches"):
            self.single_axis_block.merge_single_axis_branches()
        # SingleAxisReparamBlock内部已处理BN合并，这里无需再做额外处理
        self.norm = nn.Identity()


class SingleAxisReparamBlock(nn.Module):
    """与 DilatedReparamBlock 类似，但针对单轴并行卷积核的重参数化。 例如：并行几个1x5, 1x7, 1x9的卷积，然后合并为一个大的1x9卷积。.
    """

    def __init__(self, channels, kernel_sizes, is_vertical=True, deploy=False, attempt_use_lk_impl=True):
        super().__init__()
        self.attempt_use_lk_impl = attempt_use_lk_impl
        self.is_vertical = is_vertical
        # 假设 kernel_sizes 已经是升序的，如 [5, 7, 9]
        # 如果不是，可以先排序
        kernel_sizes = sorted(kernel_sizes)
        self.kernel_sizes = kernel_sizes

        # 确定最大核大小
        max_kernel = kernel_sizes[-1]

        # 主干卷积使用最大kernel
        if self.is_vertical:
            # 纵向 1 x max_kernel
            self.lk_origin = get_conv2d_uni(
                channels,
                channels,
                kernel_size=(1, max_kernel),
                stride=1,
                padding=(0, max_kernel // 2),
                dilation=1,
                groups=channels,
                bias=deploy,
                attempt_use_lk_impl=attempt_use_lk_impl,
            )
        else:
            # 横向 max_kernel x 1
            self.lk_origin = get_conv2d_uni(
                channels,
                channels,
                kernel_size=(max_kernel, 1),
                stride=1,
                padding=(max_kernel // 2, 0),
                dilation=1,
                groups=channels,
                bias=deploy,
                attempt_use_lk_impl=attempt_use_lk_impl,
            )

        # 对其他较小kernel_size的分支进行定义
        # 如果deploy为False，则每个分支有独立的BN
        if not deploy:
            self.origin_bn = get_bn(channels)
            # 除了主干核（即最大kernel）外，其余kernel
            # 我们把所有kernel都作为分支，包括origin本身，以保持与DilatedReparamBlock一致的流程
            # 这里主干lk_origin相当于其中一个分支，也是最大核
            self.branches = nn.ModuleList()
            self.branch_bns = nn.ModuleList()
            # 包含最大kernel本身在内，因此对kernel_sizes全部遍历
            # 但第一个分支我们将作为origin核（已经定义为lk_origin），故这里从第二个开始
            for k in kernel_sizes:
                if k == max_kernel:
                    # 这就是主干分支，不需要重复定义，只额外加bn
                    # 为了处理统一，这里直接在merge时使用origin_bn即可
                    continue
                if self.is_vertical:
                    conv = nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=(1, k),
                        stride=1,
                        padding=(0, k // 2),
                        dilation=1,
                        groups=channels,
                        bias=False,
                    )
                else:
                    conv = nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=(k, 1),
                        stride=1,
                        padding=(k // 2, 0),
                        dilation=1,
                        groups=channels,
                        bias=False,
                    )
                bn = get_bn(channels)
                self.branches.append(conv)
                self.branch_bns.append(bn)
        else:
            # 部署模式
            # 不需要分支
            pass

    def forward(self, x):
        if not hasattr(self, "origin_bn"):
            # deploy模式
            return self.lk_origin(x)
        else:
            # 非deploy模式，将所有并行分支的结果相加
            out = self.origin_bn(self.lk_origin(x))
            # 遍历分支
            # 注意，在__init__中：lk_origin相当于最大kernel分支，
            # 其他kernel的分支存放在self.branches中
            for conv, bn, k in zip(self.branches, self.branch_bns, self.kernel_sizes[:-1]):
                out = out + bn(conv(x))
            return out

    def merge_single_axis_branches(self):
        """类似merge_dilated_branches，将所有分支融合到lk_origin中."""
        if hasattr(self, "origin_bn"):
            # 主干融合
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            # 分支融合
            for conv, bn in zip(self.branches, self.branch_bns):
                branch_k, branch_b = fuse_bn(conv, bn)
                # 将branch_k对齐并合并到origin_k中
                origin_k = center_merge_kernel(origin_k, branch_k)
                origin_b += branch_b

            # 用合并后的kernel和bias创建新的单一卷积
            out_channels, _in_channels, h, w = origin_k.size()
            # h和w中有一个维度应为1（单轴卷积）
            # 确定final kernel shape
            if self.is_vertical:
                final_kernel_size = (1, w)
                final_padding = (0, w // 2)
            else:
                final_kernel_size = (h, 1)
                final_padding = (h // 2, 0)

            merged_conv = get_conv2d_uni(
                out_channels,
                out_channels,
                final_kernel_size,
                stride=1,
                padding=final_padding,
                dilation=1,
                groups=out_channels,
                bias=True,
                attempt_use_lk_impl=self.attempt_use_lk_impl,
            )
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b

            self.lk_origin = merged_conv
            # 删除BN和分支
            self.__delattr__("origin_bn")
            self.branches = nn.ModuleList()
            self.branch_bns = nn.ModuleList()


def center_merge_kernel(large_kernel, small_kernel):
    """将一个较小的核居中对齐到large_kernel大小的核上，并相加。 假设 large_kernel 和 small_kernel 都是 (C, 1, H, W) 或 (C, C, H, W) 其中 H 和 W
    是卷积核的高和宽，C 是通道数.
    """
    large_kernel.size(0)  # 获取通道数
    large_H, large_W = large_kernel.size(2), large_kernel.size(3)  # 获取 large_kernel 的 H 和 W
    small_H, small_W = small_kernel.size(2), small_kernel.size(3)  # 获取 small_kernel 的 H 和 W

    # 如果是纵向卷积（H=1），则 W 为卷积核的宽度，假设是1xK卷积核
    if small_H == 1:
        # 纵向卷积，计算在 W 维度上的填充
        pad_left = (large_W - small_W) // 2
        padded_small = F.pad(small_kernel, (pad_left, large_W - small_W - pad_left, 0, 0))
    else:
        # 横向卷积，计算在 H 维度上的填充
        pad_top = (large_H - small_H) // 2
        padded_small = F.pad(small_kernel, (0, 0, pad_top, large_H - small_H - pad_top))

    # 返回合并后的卷积核
    return large_kernel + padded_small
