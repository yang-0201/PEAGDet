import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
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


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class LearnableCoefficient(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        return x * self.bias


class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=0.1, resid_pdrop=0.1, vit_param=[], d_vit=512):
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
        self.que_proj_vis = nn.Linear(d_model, h * self.d_k)  # query projection (保持名字以兼容state_dict)
        self.key_proj_vis = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj_vis = nn.Linear(d_model, h * self.d_v)  # value projection

        self.que_proj_ir = nn.Linear(d_model, h * self.d_k)
        self.key_proj_ir = nn.Linear(d_model, h * self.d_k)
        self.val_proj_ir = nn.Linear(d_model, h * self.d_v)

        self.out_proj_vis = nn.Linear(h * self.d_v, d_model)  # output projection
        self.out_proj_ir = nn.Linear(h * self.d_v, d_model)

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
        """
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),.

        Return:
            output (tensor): dim:(b_s, nx, c).
        """
        # N/A 双分支：x[0] 为 N 分支特征，x[1] 为 A 分支特征
        N_fea_flat = x[0]
        A_fea_flat = x[1]
        b_s, nq = N_fea_flat.shape[:2]
        nk = N_fea_flat.shape[1]

        # Self-Attention（各自分支）
        N_fea_flat = self.LN1(N_fea_flat)
        q_N = self.que_proj_vis(N_fea_flat).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3).contiguous()
        k_N = self.key_proj_vis(N_fea_flat).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1).contiguous()
        v_N = self.val_proj_vis(N_fea_flat).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3).contiguous()

        A_fea_flat = self.LN2(A_fea_flat)
        q_A = self.que_proj_ir(A_fea_flat).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3).contiguous()
        k_A = self.key_proj_ir(A_fea_flat).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1).contiguous()
        v_A = self.val_proj_ir(A_fea_flat).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3).contiguous()

        att_N = torch.matmul(q_N, k_N) / np.sqrt(self.d_k)
        att_A = torch.matmul(q_A, k_A) / np.sqrt(self.d_k)

        att_N = self.attn_drop(torch.softmax(att_N, -1))
        att_A = self.attn_drop(torch.softmax(att_A, -1))

        out_N = torch.matmul(att_N, v_N).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out_N = self.resid_drop(self.out_proj_vis(out_N))

        out_A = torch.matmul(att_A, v_A).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out_A = self.resid_drop(self.out_proj_ir(out_A))

        return [out_N, out_A]


class CrossAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=0.1, resid_pdrop=0.1, vit_param=[], d_vit=512):
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

        # key, query, value projections for all heads (保持属性名不变以兼容已有权重)
        self.que_proj_vis = nn.Linear(d_model, h * self.d_k)
        self.key_proj_vis = nn.Linear(d_model, h * self.d_k)
        self.val_proj_vis = nn.Linear(d_model, h * self.d_v)

        self.que_proj_ir = nn.Linear(d_model, h * self.d_k)
        self.key_proj_ir = nn.Linear(d_model, h * self.d_k)
        self.val_proj_ir = nn.Linear(d_model, h * self.d_v)

        self.out_proj_vis = nn.Linear(h * self.d_v, d_model)
        self.out_proj_ir = nn.Linear(h * self.d_v, d_model)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # layer norm
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        self.init_weights()

        if vit_param:
            self.que_proj_vis.weight.data[:d_vit, :d_vit] = vit_param[0]
            self.key_proj_vis.weight.data[:d_vit, :d_vit] = vit_param[1]
            self.val_proj_vis.weight.data[:d_vit, :d_vit] = vit_param[2]

            self.que_proj_ir.weight.data[:d_vit, :d_vit] = vit_param[0]
            self.key_proj_ir.weight.data[:d_vit, :d_vit] = vit_param[1]
            self.val_proj_ir.weight.data[:d_vit, :d_vit] = vit_param[2]

            self.que_proj_vis.bias.data[:d_vit] = vit_param[3]
            self.key_proj_vis.bias.data[:d_vit] = vit_param[4]
            self.val_proj_vis.bias.data[:d_vit] = vit_param[5]

            self.que_proj_vis.bias.data[:d_vit] = vit_param[3]
            self.key_proj_ir.bias.data[:d_vit] = vit_param[4]
            self.val_proj_ir.bias.data[:d_vit] = vit_param[5]

            self.out_proj_vis.weight.data[:d_vit, :d_vit] = vit_param[6]
            self.out_proj_ir.weight.data[:d_vit, :d_vit] = vit_param[6]

            self.out_proj_vis.bias.data[:d_vit] = vit_param[7]
            self.out_proj_ir.bias.data[:d_vit] = vit_param[7]

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
        """Cross-Attention: N 分支使用 A 的键值，A 分支使用 N 的键值."""
        N_fea_flat = x[0]
        A_fea_flat = x[1]
        b_s, nq = N_fea_flat.shape[:2]
        nk = N_fea_flat.shape[1]

        N_fea_flat = self.LN1(N_fea_flat)
        q_N = self.que_proj_vis(N_fea_flat).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3).contiguous()
        k_N = self.key_proj_vis(N_fea_flat).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1).contiguous()
        v_N = self.val_proj_vis(N_fea_flat).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3).contiguous()

        A_fea_flat = self.LN2(A_fea_flat)
        q_A = self.que_proj_ir(A_fea_flat).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3).contiguous()
        k_A = self.key_proj_ir(A_fea_flat).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1).contiguous()
        v_A = self.val_proj_ir(A_fea_flat).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3).contiguous()

        # 交叉注意力
        att_N = torch.matmul(q_A, k_N) / np.sqrt(self.d_k)  # N 查询 A？（保持与原逻辑一致：q_ir@k_vis）
        att_A = torch.matmul(q_N, k_A) / np.sqrt(self.d_k)  # A 查询 N？（保持与原逻辑一致：q_vis@k_ir）

        att_N = self.attn_drop(torch.softmax(att_N, -1))
        att_A = self.attn_drop(torch.softmax(att_A, -1))

        out_N = torch.matmul(att_N, v_N).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out_N = self.resid_drop(self.out_proj_vis(out_N))

        out_A = torch.matmul(att_A, v_A).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out_A = self.resid_drop(self.out_proj_ir(out_A))

        return [out_N, out_A]


class CrossTransformerBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, loops_num=1, vit_layer=9):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)
        """
        super().__init__()
        self.loops = loops_num

        p = []
        d_vit = 384
        if vit_layer not in [0, 1, 2, 3, 4, 5]:
            d_vit = 512

        self.selfatt = SelfAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop, p, d_vit=d_vit)
        self.crossatt = CrossAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop, p, d_vit=d_vit)

        self.mlp_vis = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            nn.GELU(),
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )
        self.mlp_ir = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            nn.GELU(),
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )

        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

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

        for _ in range(self.loops):
            N_fea_flat, A_fea_flat = self.selfatt([N_fea_flat, A_fea_flat])
            N_fea_out, A_fea_out = self.crossatt([N_fea_flat, A_fea_flat])

            N_att_out = self.coefficient1(N_fea_flat) + self.coefficient2(N_fea_out)
            A_att_out = self.coefficient3(A_fea_flat) + self.coefficient4(A_fea_out)

            N_fea_flat = self.coefficient5(N_att_out) + self.coefficient6(self.mlp_vis(self.LN2(N_att_out)))
            A_fea_flat = self.coefficient7(A_att_out) + self.coefficient8(self.mlp_ir(self.LN2(A_att_out)))

        return [N_fea_flat, A_fea_flat]


class LearnableWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.tensor([0.5], dtype=torch.float32), requires_grad=True)
        self.w2 = nn.Parameter(torch.tensor([0.5], dtype=torch.float32), requires_grad=True)

    def forward(self, x1, x2):
        return x1 * self.w1 + x2 * self.w2


class AdaptivePool2d(nn.Module):
    def __init__(self, output_h, output_w, pool_type="avg"):
        super().__init__()
        self.output_h = output_h
        self.output_w = output_w
        self.pool_type = pool_type.lower()
        assert self.pool_type in ["avg", "max"], "pool_type must be 'avg' or 'max'"

    def forward(self, x):
        bs, c, input_h, input_w = x.shape

        # 情况1：输入尺寸 ≤ 输出尺寸 → 直接返回或上采样
        if input_h <= self.output_h or input_w <= self.output_w:
            if input_h != self.output_h or input_w != self.output_w:
                return F.interpolate(x, size=(self.output_h, self.output_w), mode="bilinear", align_corners=False)
            return x

        # 情况2：输入尺寸 > 输出尺寸 → 动态计算池化参数
        stride_h = max(1, input_h // self.output_h)
        stride_w = max(1, input_w // self.output_w)
        kernel_size = (
            input_h - (self.output_h - 1) * stride_h,
            input_w - (self.output_w - 1) * stride_w,
        )

        if self.pool_type == "avg":
            return nn.AvgPool2d(kernel_size=kernel_size, stride=(stride_h, stride_w), padding=0)(x)
        else:
            return nn.MaxPool2d(kernel_size=kernel_size, stride=(stride_h, stride_w), padding=0)(x)


class PCAF(nn.Module):
    def __init__(
        self,
        in_model,
        d_model,
        vert_anchors=16,
        horz_anchors=16,
        h=8,
        block_exp=4,
        n_layer=3,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
    ):
        super().__init__()
        print("n_lyer: ", n_layer)
        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        d_k = d_model
        d_v = d_model

        self.pos_emb_vis = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
        self.pos_emb_ir = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))

        self.avgpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, "avg")
        self.maxpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, "max")

        # LearnableCoefficient
        self.vis_coefficient = LearnableWeights()
        self.ir_coefficient = LearnableWeights()

        # init weights
        self.apply(self._init_weights)

        vit_layer = -1
        if vert_anchors == 20:
            vit_layer = 0
        elif vert_anchors == 16:
            vit_layer = 6
        elif vert_anchors == 10:
            vit_layer = 6

        # cross transformer
        self.crosstransformer = nn.Sequential(
            *[
                CrossTransformerBlock(
                    d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, vit_layer=vit_layer + layer
                )
                for layer in range(n_layer)
            ]
        )

        # Concat
        self.concat = Concat(dimension=1)

        # conv1x1
        self.conv1x1_in1 = Conv(c1=in_model, c2=d_model, k=1, s=1, p=0, g=1, act=True)
        self.conv1x1_in2 = Conv(c1=in_model, c2=d_model, k=1, s=1, p=0, g=1, act=True)
        self.conv1x1_out = Conv(c1=d_model * 2, c2=d_model, k=1, s=1, p=0, g=1, act=True)

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
        # 将输入两路特征映射到 d_model 通道
        N_fea = self.conv1x1_in1(x[0])
        A_fea = self.conv1x1_in2(x[1])

        assert N_fea.shape[0] == A_fea.shape[0]
        bs, c, h, w = N_fea.shape

        # 自适应池化 + 加权融合（avg/max）
        new_N_fea = self.vis_coefficient(self.avgpool(N_fea), self.maxpool(N_fea))
        new_c, new_h, new_w = new_N_fea.shape[1], new_N_fea.shape[2], new_N_fea.shape[3]

        N_fea_flat = new_N_fea.view(bs, new_c, -1).permute(0, 2, 1).contiguous() + self.pos_emb_vis
        new_A_fea = self.ir_coefficient(self.avgpool(A_fea), self.maxpool(A_fea))
        A_fea_flat = new_A_fea.view(bs, new_c, -1).permute(0, 2, 1).contiguous() + self.pos_emb_ir

        # 经过多层 Cross-Transformer
        N_fea_flat, A_fea_flat = self.crosstransformer([N_fea_flat, A_fea_flat])

        # 还原为 (B, C, H, W)
        N_fea_CFE = N_fea_flat.view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2).contiguous()
        if self.training:
            N_fea_CFE = F.interpolate(N_fea_CFE, size=(h, w), mode="nearest")
        else:
            N_fea_CFE = F.interpolate(N_fea_CFE, size=(h, w), mode="bilinear")
        new_N_fea = N_fea_CFE + N_fea

        A_fea_CFE = A_fea_flat.view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2).contiguous()
        if self.training:
            A_fea_CFE = F.interpolate(A_fea_CFE, size=(h, w), mode="nearest")
        else:
            A_fea_CFE = F.interpolate(A_fea_CFE, size=(h, w), mode="bilinear")
        new_A_fea = A_fea_CFE + A_fea

        # 融合两路并回到 d_model
        new_fea = self.concat([new_N_fea, new_A_fea])
        new_fea = self.conv1x1_out(new_fea)
        return new_fea


class Feature_Pool(nn.Module):
    def __init__(self, dim, ratio=2):
        super().__init__()
        self.gap_pool = nn.AdaptiveAvgPool2d(1)
        self.down = nn.Linear(dim, dim * ratio)
        self.act = nn.GELU()
        self.up = nn.Linear(dim * ratio, dim)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.up(self.act(self.down(self.gap_pool(x).permute(0, 2, 3, 1)))).permute(0, 3, 1, 2).view(b, c)
        return y


class Channel_Attention(nn.Module):
    def __init__(self, dim, ratio=16):
        super().__init__()
        self.gap_pool = nn.AdaptiveMaxPool2d(1)
        self.down = nn.Linear(dim, dim // ratio)
        self.act = nn.GELU()
        self.up = nn.Linear(dim // ratio, dim)

    def forward(self, x):
        max_out = self.up(self.act(self.down(self.gap_pool(x).permute(0, 2, 3, 1)))).permute(0, 3, 1, 2)
        return max_out


class Spatial_Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, 1, kernel_size=1, bias=True)

    def forward(self, x):
        x1 = self.conv1(x)
        return x1


class DPAG(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp_pool = Feature_Pool(dim)
        self.dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=7, padding=3, groups=dim)
        self.ecse = Channel_Attention(dim * 2)
        self.ccse = Channel_Attention(dim)  # 保留以兼容原结构（此处未使用）
        self.sse_N = Spatial_Attention(dim)  # 原 sse_r
        self.sse_A = Spatial_Attention(dim)  # 原 sse_t

    def forward(self, x1, x2):
        # 统一命名：N（例如 Non-contrast）与 A（例如 Arterial）
        N, A = x1, x2
        b, c, h, w = N.size()

        # 全局/通道聚合 + L2 归一化
        N_y = self.mlp_pool(N)  # 期望形状 [b, c]
        A_y = self.mlp_pool(A)  # 期望形状 [b, c]
        N_y = N_y / (N_y.norm(dim=1, keepdim=True) + 1e-6)
        A_y = A_y / (A_y.norm(dim=1, keepdim=True) + 1e-6)

        # 构造通道间相似度（逐通道对齐得到对角）
        N_y = N_y.view(b, c, 1)  # [b, c, 1]
        A_y = A_y.view(b, 1, c)  # [b, 1, c]
        logits_per = c * (N_y @ A_y)  # [b, c, c]
        cross_gate = torch.diagonal(torch.sigmoid(logits_per), dim1=-2, dim2=-1).view(b, c, 1, 1)  # [b, c, 1, 1]
        add_gate = 1.0 - cross_gate

        # 交换(e)/补充(c)两条路径
        New_N_e = N * cross_gate
        New_A_e = A * cross_gate
        New_N_c = N * add_gate
        New_A_c = A * add_gate

        # 交换路径融合：DWConv + 通道注意力
        x_cat_e = torch.cat((New_N_e, New_A_e), dim=1)  # [b, 2c, h, w]
        fuse_gate_e = torch.sigmoid(self.ecse(self.dwconv(x_cat_e)))  # 期望 [b, 2c, *, *] 或 [b, 2c, 1, 1]
        N_gate_e, A_gate_e = fuse_gate_e[:, :c, ...], fuse_gate_e[:, c : 2 * c, ...]

        # 门控融合：交换 + 补充
        New_N = New_N_e * N_gate_e + New_N_c
        New_A = New_A_e * A_gate_e + New_A_c

        # 分支级空间注意力 + softmax 归一耦合
        New_fuse_N = self.sse_N(New_N)  # [b, 1, h, w]
        New_fuse_A = self.sse_A(New_A)  # [b, 1, h, w]
        attention_vector = torch.softmax(torch.cat([New_fuse_N, New_fuse_A], dim=1), dim=1)  # [b, 2, h, w]
        att_N, att_A = attention_vector[:, :1, ...], attention_vector[:, 1:2, ...]

        New_N = New_N * att_N
        New_A = New_A * att_A
        # New_fuse = New_N + New_A

        return New_N, New_A


class PCAF_DPAG(nn.Module):
    def __init__(
        self,
        in_model,
        d_model,
        vert_anchors=16,
        horz_anchors=16,
        h=8,
        block_exp=4,
        n_layer=3,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
    ):
        super().__init__()
        print("n_lyer: ", n_layer)
        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        d_k = d_model
        d_v = d_model

        self.pos_emb_vis = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
        self.pos_emb_ir = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))

        self.avgpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, "avg")
        self.maxpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, "max")

        # LearnableCoefficient
        self.vis_coefficient = LearnableWeights()
        self.ir_coefficient = LearnableWeights()

        # init weights
        self.apply(self._init_weights)

        vit_layer = -1
        if vert_anchors == 20:
            vit_layer = 0
        elif vert_anchors == 16:
            vit_layer = 6
        elif vert_anchors == 10:
            vit_layer = 6

        # cross transformer
        self.crosstransformer = nn.Sequential(
            *[
                CrossTransformerBlock(
                    d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, vit_layer=vit_layer + layer
                )
                for layer in range(n_layer)
            ]
        )

        # Concat
        self.concat = Concat(dimension=1)

        # conv1x1
        self.conv1x1_in1 = Conv(c1=in_model, c2=d_model, k=1, s=1, p=0, g=1, act=True)
        self.conv1x1_in2 = Conv(c1=in_model, c2=d_model, k=1, s=1, p=0, g=1, act=True)
        self.conv1x1_out = Conv(c1=d_model * 2, c2=d_model, k=1, s=1, p=0, g=1, act=True)

        self.dpag = DPAG(d_model)

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
        # N, A = self.dpag(x[0], x[1])

        # 将输入两路特征映射到 d_model 通道
        N_fea = self.conv1x1_in1(x[0])
        A_fea = self.conv1x1_in2(x[1])
        N_fea, A_fea = self.dpag(N_fea, A_fea)

        assert N_fea.shape[0] == A_fea.shape[0]
        bs, c, h, w = N_fea.shape

        # 自适应池化 + 加权融合（avg/max）
        new_N_fea = self.vis_coefficient(self.avgpool(N_fea), self.maxpool(N_fea))
        new_c, new_h, new_w = new_N_fea.shape[1], new_N_fea.shape[2], new_N_fea.shape[3]

        N_fea_flat = new_N_fea.view(bs, new_c, -1).permute(0, 2, 1).contiguous() + self.pos_emb_vis
        new_A_fea = self.ir_coefficient(self.avgpool(A_fea), self.maxpool(A_fea))
        A_fea_flat = new_A_fea.view(bs, new_c, -1).permute(0, 2, 1).contiguous() + self.pos_emb_ir

        # 经过多层 Cross-Transformer
        N_fea_flat, A_fea_flat = self.crosstransformer([N_fea_flat, A_fea_flat])

        # 还原为 (B, C, H, W)
        N_fea_CFE = N_fea_flat.view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2).contiguous()
        if self.training:
            N_fea_CFE = F.interpolate(N_fea_CFE, size=(h, w), mode="nearest")
        else:
            N_fea_CFE = F.interpolate(N_fea_CFE, size=(h, w), mode="bilinear")
        new_N_fea = N_fea_CFE + N_fea

        A_fea_CFE = A_fea_flat.view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2).contiguous()
        if self.training:
            A_fea_CFE = F.interpolate(A_fea_CFE, size=(h, w), mode="nearest")
        else:
            A_fea_CFE = F.interpolate(A_fea_CFE, size=(h, w), mode="bilinear")
        new_A_fea = A_fea_CFE + A_fea

        # 融合两路并回到 d_model
        new_fea = self.concat([new_N_fea, new_A_fea])
        new_fea = self.conv1x1_out(new_fea)
        # new_fea = self.dpag(new_N_fea, new_A_fea)
        return new_fea
