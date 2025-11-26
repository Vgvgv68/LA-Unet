import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm.layers.weight_init as weight_init
from safetensors.torch import load_file
import torchvision
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision import models



class Activation(nn.ReLU):
    def __init__(self, dim, act_num=3, deploy=False):
        super(Activation, self).__init__()
        self.deploy = deploy
        self.weight = torch.nn.Parameter(torch.randn(dim, 1, act_num * 2 + 1, act_num * 2 + 1))
        self.bias = None
        self.bn = nn.BatchNorm2d(dim, eps=1e-6)
        self.dim = dim
        self.act_num = act_num
        weight_init.trunc_normal_(self.weight, std=.02)

    def forward(self, x):
        if self.deploy:
            return torch.nn.functional.conv2d(
                super(Activation, self).forward(x),
                self.weight, self.bias, padding=(self.act_num * 2 + 1) // 2, groups=self.dim)
        else:
            return self.bn(torch.nn.functional.conv2d(
                super(Activation, self).forward(x),
                self.weight, padding=self.act_num, groups=self.dim))

    def _fuse_bn_tensor(self, weight, bn):
        kernel = weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (0 - running_mean) * gamma / std

class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class DynamicChannelAttention(nn.Module):
    def __init__(self, dim, reduction_ratio=4, groups=2):
        super().__init__()
        self.groups = groups
        assert dim % groups == 0, "dim must be divisible by groups"

        # 分组后的维度
        g_dim = dim // groups

        # 共享的通道注意力模块（只处理第一个分组）
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(g_dim, g_dim // reduction_ratio, 1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(g_dim // reduction_ratio, g_dim, 1),
            nn.Sigmoid()
        )

        # 共享的空间注意力模块（只处理第一个分组）
        self.depth_conv = nn.Conv2d(g_dim, g_dim, 3,
                                    padding=1, groups=g_dim)

        # 参数复用：使用同一组参数处理所有分组的特征
        self.shared_transform = nn.Sequential(
            nn.Conv2d(g_dim, g_dim, 1),  # 共享的变换层
            nn.ReLU6(inplace=True)
        )

        # 自适应融合参数
        self.fuse = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, x):
        # 将特征通道分成多个组
        split_x = torch.chunk(x, self.groups, dim=1)

        # 第一个分组生成注意力
        g0 = split_x[0]
        ca = self.channel_att(g0)
        sa = self.depth_conv(g0)

        # 自适应融合注意力
        att = ca + self.fuse * torch.sigmoid(sa)

        # 对每个分组应用共享变换和注意力
        processed = []
        for i, g in enumerate(split_x):
            # 参数复用：所有分组共享相同的变换层
            trans_g = self.shared_transform(g)

            # 应用统一的注意力（来自第一个分组）
            processed.append(trans_g * att)

        # 拼接所有分组
        return torch.cat(processed, dim=1)

    def _reset_parameters(self):
        # 初始化融合参数
        nn.init.constant_(self.fuse, 0.5)


    def __init__(self, dim, reduction_ratio=4):
        super(Attention_block, self).__init__()
        # 计算中间维度
        F_int = dim // reduction_ratio

        # 门控信号路径（使用1x1卷积降维）
        self.W_g = nn.Sequential(
            nn.Conv2d(dim, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # 输入特征路径（使用1x1卷积降维）
        self.W_x = nn.Sequential(
            nn.Conv2d(dim, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # 注意力生成路径
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 门控信号和输入特征使用相同的输入x
        g1 = self.W_g(x)
        x1 = self.W_x(x)

        # 结合门控信号和输入特征
        psi = self.relu(g1 + x1)

        # 生成注意力图
        attention_map = self.psi(psi)

        # 应用注意力机制
        return x * attention_map




    def __init__(self, channels, groups=32):
        super(EMAAttention, self).__init__()
        self.groups = groups
        assert channels // self.groups > 0, "通道数必须大于分组数"

        # 各种池化和归一化层
        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # 分组归一化和卷积层
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups,
                                 kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups,
                                 kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # 将输入特征图分组
        group_x = x.reshape(batch_size * self.groups, -1, height, width)  # [b*g, c//g, h, w]

        # 高度和宽度方向的池化
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)

        # 连接并应用1x1卷积
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [height, width], dim=2)

        # 应用分组归一化和Sigmoid激活
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)

        # 计算注意力权重
        x11 = self.softmax(self.agp(x1).reshape(batch_size * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(batch_size * self.groups, channels // self.groups, -1)  # [b*g, c//g, hw]

        x21 = self.softmax(self.agp(x2).reshape(batch_size * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(batch_size * self.groups, channels // self.groups, -1)  # [b*g, c//g, hw]

        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(batch_size * self.groups, 1, height, width)

        # 应用注意力权重并恢复原始形状
        return (group_x * weights.sigmoid()).reshape(batch_size, channels, height, width)


    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.gate_channels = gate_channels
        self.reduction_ratio = reduction_ratio
        self.pool_types = pool_types
        self.no_spatial = no_spatial

        # 通道注意力部分
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )

        # 空间注意力部分
        if not no_spatial:
            kernel_size = 7
            self.spatial_conv = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False),
                nn.BatchNorm2d(1, eps=1e-5, momentum=0.01)
            )

    def forward(self, x):
        # 通道注意力
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                pool = F.adaptive_avg_pool2d(x, 1)
            elif pool_type == 'max':
                pool = F.adaptive_max_pool2d(x, 1)
            channel_att_raw = self.mlp(pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum += channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3)
        x_out = x * scale

        # 空间注意力
        if not self.no_spatial:
            avg_out = torch.mean(x_out, dim=1, keepdim=True)
            max_out, _ = torch.max(x_out, dim=1, keepdim=True)
            spatial_compress = torch.cat([avg_out, max_out], dim=1)
            spatial_att = self.spatial_conv(spatial_compress)
            spatial_scale = torch.sigmoid(spatial_att)
            x_out = x_out * spatial_scale

        return x_out

class FeatureDistiller(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1, groups=in_channels // 4),
            nn.GELU(),
            nn.Conv2d(in_channels // 4, in_channels, 1)
        )
        self.conv[0].weight = nn.Parameter(torch.ones_like(self.conv[0].weight))

    def forward(self, x):
        identity = x
        return identity + 0.2 * self.conv(x)

class AdaptiveSpatialRegularization(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv = nn.Conv2d(channel, 1, 3, padding=1)
        # 参数共享策略
        self.conv.weight = nn.Parameter(torch.ones_like(self.conv.weight) * 0.01)

    def forward(self, x):
        mask = torch.sigmoid(self.conv(x))
        return x * (1 + mask)

class Block(nn.Module):
    def __init__(self, dim, dim_out, act_num=3, stride=2, deploy=False, groups=4):
        super().__init__()
        self.act_learn = 0
        self.deploy = deploy
        self.groups = groups
        c_dim = dim // groups

        # 空间共享参数初始化
        self.share_space = nn.ParameterList([
            nn.Parameter(torch.Tensor(1, c_dim, 8, 8)) for _ in range(groups)
        ])
        for param in self.share_space:
            nn.init.ones_(param)

        # 分组卷积处理
        self.group_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c_dim, c_dim, 3, padding=1, groups=c_dim),
                nn.GELU(),
                nn.Conv2d(c_dim, c_dim, 1)
            ) for _ in range(groups)
        ])

        # 通道融合层
        self.norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')

        # 深度可分离卷积
        self.ldw = nn.Sequential(
            nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim_out, 1),
        )

        # 下采样层
        self.att = Attention_block(dim_out)
        self.pool = nn.Identity() if stride == 1 else nn.MaxPool2d(stride)
        self.act = Activation(dim_out, act_num, deploy=self.deploy)

    def forward(self, x):
        # 分组空间变换
        x = self.norm(x)
        group_x = torch.chunk(x, self.groups, dim=1)
        processed = []
        for i in range(self.groups):
            spatial = F.interpolate(self.share_space[i], size=x.shape[2:], mode='bilinear')
            processed.append(group_x[i] * self.group_convs[i](spatial))

        # 通道重排
        x = torch.cat([processed[(i + 1) % self.groups] for i in range(self.groups)], dim=1)

        # 深度可分离卷积
        x = self.ldw(x)
        x = self.att(x)
        x = self.pool(x)

        return self.act(x)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            weight_init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.weight
        bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)

        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std

class UpBlock(nn.Module):
    def __init__(self, dim, dim_out, act_num=3, factor=2, deploy=False, groups=4):
        super().__init__()
        self.act_learn = 0
        self.deploy = deploy
        self.groups = groups
        c_dim = dim // groups

        # 空间共享参数初始化
        self.share_space = nn.ParameterList([
            nn.Parameter(torch.Tensor(1, c_dim, 8, 8)) for _ in range(groups)
        ])
        for param in self.share_space:
            nn.init.ones_(param)

        # 分组卷积处理
        self.group_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c_dim, c_dim, 3, padding=1, groups=c_dim),
                nn.GELU(),
                nn.Conv2d(c_dim, c_dim, 1)
            ) for _ in range(groups)
        ])

        # 通道融合层
        self.norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')

        # 深度可分离卷积
        self.ldw = nn.Sequential(
            nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim_out, 1),
        )

        # 上采样层
        self.upsample = nn.Upsample(scale_factor=factor, mode='bilinear')
        self.act = Activation(dim_out, act_num, deploy=self.deploy)
        self.spatial_reg = AdaptiveSpatialRegularization(dim_out)

    def forward(self, x):
        # 分组空间变换
        x = self.norm(x)
        group_x = torch.chunk(x, self.groups, dim=1)
        processed = []
        for i in range(self.groups):
            spatial = F.interpolate(self.share_space[i], size=x.shape[2:], mode='bilinear')
            processed.append(group_x[i] * self.group_convs[i](spatial))

        # 通道重排
        x = torch.cat([processed[(i + 1) % self.groups] for i in range(self.groups)], dim=1)

        # 深度可分离卷积
        x = self.ldw(x)
        x = self.spatial_reg(x)
        x = self.upsample(x)

        return self.act(x)

    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.weight
        bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)

        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std

class LA_UNet(nn.Module):
    def __init__(self, in_chans=3, dims=[80, 128, 256, 512], dims2=[80, 40, 24, 16],
                 drop_rate=0, act_num=1, strides=[2, 2, 2], deploy=False):
        super().__init__()
        self.deploy = deploy

        mobile = models.mobilenet_v3_large(pretrained=True)

        self.firstconv = mobile.features[0]
        self.encoder1 = nn.Sequential(
            mobile.features[1],
            mobile.features[2],
        )
        self.encoder2 = nn.Sequential(
            mobile.features[3],
            mobile.features[4],
            mobile.features[5],
        )
        self.encoder3 = nn.Sequential(
            mobile.features[6],
            mobile.features[7],
            mobile.features[8],
            mobile.features[9],
            mobile.features[10],

        )

        self.act_learn = 0
        self.stages = nn.ModuleList()
        self.up_stages1 = nn.ModuleList()
        self.up_stages2 = nn.ModuleList()


        self.depth = len(strides)
        self.fuse_weights = nn.ParameterList([
            nn.Parameter(torch.Tensor(2, dims[2 - i]))
            for i in range(self.depth)
        ])
        for weight in self.fuse_weights:
            nn.init.constant_(weight, 1.)


        for i in range(len(strides)):
            self.stages.append(Block(dims[i], dims[i + 1], act_num, strides[i], deploy))

        for i in range(len(strides)):
            self.up_stages1.append(UpBlock(dims[3 - i], dims[2 - i], act_num, strides[2 - i], deploy))

        for i in range(3):
            self.up_stages2.append(UpBlock(dims2[i], dims2[i + 1], act_num, 2, deploy))

        self.final = nn.Sequential(
            UpBlock(16, 16, act_num, 2),
            nn.Conv2d(16, 1, 1)
        )

        self.distillers = nn.ModuleList([
            FeatureDistiller(dim) for dim in dims
        ])


    def _dynamic_fusion(self, dec_feat, enc_feat, index):

        alpha, beta = self.fuse_weights[index][0], self.fuse_weights[index][1]
        return (alpha[None, :, None, None] * dec_feat +
                beta[None, :, None, None] * enc_feat)

    def forward(self, x):
        # 编码阶段
        x = self.firstconv(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        # 下采样中间特征
        encoder = []
        for i in range(self.depth):
            e3 = self.distillers[i](e3)
            encoder.append(e3)
            e3 = self.stages[i](e3)

        # 上采样+动态融合
        for i in range(self.depth):
            e3 = self.up_stages1[i](e3)
            e3 = self._dynamic_fusion(e3, encoder[2 - i], i)

        # 多尺度融合
        e3 = self.up_stages2[0](e3) + e2
        e3 = self.up_stages2[1](e3) + e1
        e3 = self.up_stages2[2](e3)

        return self.final(e3)

    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.weight
        bias = conv.bias
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std
