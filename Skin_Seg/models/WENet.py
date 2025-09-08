# param：0.014 GFLOPs：0.110
import os
import time
import math
import torch
import joblib
import random
import warnings
import argparse
import numpy as np
import torchvision
import pandas as pd
from tqdm import tqdm
from glob import glob
import torch.nn as nn
import sklearn.externals
import torch.optim as optim
# from dataset import Dataset
from datetime import datetime
from skimage.io import imread
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import datasets, models, transforms


def SeedSed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SeedSed(seed=10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DepthwiseSeparableConv, self).__init__()

        # 多核逐通道卷积
        self.depth_conv1_3_1 = nn.Conv2d(in_channels=in_channel,
                                         out_channels=in_channel,
                                         kernel_size=(1, 3),
                                         stride=1,
                                         padding=(0, 1),
                                         groups=in_channel)
        self.depth_conv3_1_1 = nn.Conv2d(in_channels=in_channel,
                                         out_channels=in_channel,
                                         kernel_size=(3, 1),
                                         stride=1,
                                         padding=(1, 0),
                                         groups=in_channel)
        self.depth_conv1_3_5 = nn.Conv2d(in_channels=in_channel,
                                         out_channels=in_channel,
                                         kernel_size=(1, 3),
                                         dilation=(1, 5),
                                         stride=1,
                                         padding=(0, 5),
                                         groups=in_channel)
        self.depth_conv3_1_5 = nn.Conv2d(in_channels=in_channel,
                                         out_channels=in_channel,
                                         kernel_size=(3, 1),
                                         dilation=(5, 1),
                                         stride=1,
                                         padding=(5, 0),
                                         groups=in_channel)

    def forward(self, input):
        out1 = self.depth_conv3_1_1(self.depth_conv1_3_1(input))
        out2 = self.depth_conv3_1_5(self.depth_conv1_3_5(input))
        out = out1 + out2
        return out


class SDPC(nn.Module):
    SeedSed(seed=10)

    def __init__(self, in_channels, out_channels):
        super(SDPC, self).__init__()

        self.conv1x1_squeeze = nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=1,
                                         stride=1, padding=0)
        self.conv1 = DepthwiseSeparableConv(out_channels // 2, out_channels // 2)
        self.conv2 = DepthwiseSeparableConv(out_channels // 2, out_channels // 2)
        self.conv1x1_excitation = nn.Conv2d(in_channels=out_channels // 2, out_channels=out_channels, kernel_size=1,
                                            stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)

    def forward(self, x):
        x = self.conv1x1_squeeze(x)
        x = F.gelu(self.bn1(self.conv2(self.conv1(x))))
        x = self.conv1x1_excitation(x)

        return x


# Statistical Multi-feature Adaptive spatial Recalibration Attention(SASA)
class SASA(nn.Module):
    SeedSed(seed=10)

    def __init__(self, in_channels):
        super(SASA, self).__init__()

        self.weight = nn.Parameter(torch.ones(4))

        self.conv = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU()  # 增强非线性
        )

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x_std = torch.std(x, dim=1, keepdim=True)
        x_energy = torch.mean(x ** 2, dim=1, keepdim=True)
        x_cat = torch.cat([x_avg, x_max, x_std, x_energy], dim=1)

        weights = F.softmax(self.weight, dim=0)  # 归一化权重
        x_weighted = (x_cat * weights.view(1, 4, 1, 1)).sum(dim=1, keepdim=True)  # 加权融合

        x_attention = torch.sigmoid(self.conv(x_weighted))

        return x * x_attention


# Statistical Multi-feature Adaptive Channel Recalibration Attention(SACA)
class saca(nn.Module):
    def __init__(self, channel, kernel_size=3):
        super().__init__()

        # 多特征权重学习
        self.weights = nn.Parameter(torch.ones(4))  # 均值/最大值/标准差/能量
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        b, c, h, w = x.shape

        # 多维度通道统计特征 --------------------------------
        # 均值池化
        x_avg = x.mean(dim=[2, 3], keepdim=True)  # [B,C,1,1]
        # 最大值池化
        x_max = x.amax(dim=[2, 3], keepdim=True)  # [B,C,1,1]
        # 标准差池化
        x_std = torch.std(x, dim=[2, 3], keepdim=True)  # [B,C,1,1]
        # 能量池化 (L2 Norm)
        x_energy = (x ** 2).mean(dim=[2, 3], keepdim=True)  # [B,C,1,1]

        # 自适应特征融合 --------------------------------
        weights = F.softmax(self.weights, dim=0)  # 归一化权重
        fused = (weights[0] * x_avg + weights[1] * x_max +
                 weights[2] * x_std + weights[3] * x_energy)  # [B,C,1,1]

        x_c = fused.squeeze(-1).permute(0, 2, 1)  # 移除最后一个维度并转置，为1D卷积准备，变为bs,1,c
        x_c = self.conv(x_c)  # 对转置后的y应用1D卷积，得到bs,1,c维度的输出
        x_c = torch.sigmoid(x_c)  # 应用Sigmoid函数激活，得到最终的注意力权重
        x_c = x_c.permute(0, 2, 1).unsqueeze(-1)  # 再次转置并增加一个维度，以匹配原始输入x的维度
        return x * x_c.expand_as(x)


class Downsample_block(nn.Module):
    SeedSed(seed=10)

    def __init__(self, in_channels, out_channels, self_is=True):
        super(Downsample_block, self).__init__()
        self.is_down = self_is

        self.sdpc = SDPC(in_channels, out_channels)

        # SACA
        self.ca = saca(out_channels)

    def forward(self, x):

        y = self.sdpc(x)
        if self.is_down:
            y = self.ca(y)
            x = F.max_pool2d(y, 2, stride=2)
            return x, y
        else:
            return y


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


class Light_Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Light_Up, self).__init__()
        # 多核逐通道卷积

        x = 8
        y = 8
        k_size = 3
        pad = (k_size - 1) // 2

        self.share_memory1 = nn.Parameter(torch.Tensor(1, out_channels // 4, x, y), requires_grad=True)
        nn.init.ones_(self.share_memory1)
        self.xy_conv1 = nn.Sequential(
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=k_size, padding=pad, groups=out_channels // 4),
            nn.Conv2d((out_channels // 4), (out_channels // 4), 1))

        self.share_memory2 = nn.Parameter(torch.Tensor(1, 1, out_channels // 4, x), requires_grad=True)
        nn.init.ones_(self.share_memory2)
        self.zx_conv2 = nn.Sequential(
            nn.Conv1d(out_channels // 4, out_channels // 4, kernel_size=k_size, padding=pad, groups=out_channels // 4),
            nn.Conv1d(out_channels // 4, out_channels // 4, 1))

        self.share_memory3 = nn.Parameter(torch.Tensor(1, 1, out_channels // 4, y), requires_grad=True)
        nn.init.ones_(self.share_memory3)
        self.zy_conv3 = nn.Sequential(
            nn.Conv1d(out_channels // 4, out_channels // 4, kernel_size=k_size, padding=pad, groups=out_channels // 4),
            nn.Conv1d(out_channels // 4, out_channels // 4, 1))

        self.conv1x1_1 = SDPC(in_channels // 4, out_channels // 4)
        self.conv1x1_2 = SDPC(in_channels // 4, out_channels // 4)
        self.conv1x1_3 = SDPC(in_channels // 4, out_channels // 4)
        self.conv1x1_4 = SDPC(in_channels // 4, out_channels // 4)

        # self.conv1x1_1 = nn.Sequential(
        #     nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=k_size, padding=pad, groups=in_channels // 4),
        #     nn.Conv2d((in_channels // 4), (out_channels // 4), 1))
        # self.conv1x1_2 = nn.Sequential(
        #     nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=k_size, padding=pad, groups=in_channels // 4),
        #     nn.Conv2d((in_channels // 4), (out_channels // 4), 1))
        # self.conv1x1_3 = nn.Sequential(
        #     nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=k_size, padding=pad, groups=in_channels // 4),
        #     nn.Conv2d((in_channels // 4), (out_channels // 4), 1))
        # self.conv1x1_4 = nn.Sequential(
        #     nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=k_size, padding=pad, groups=in_channels // 4),
        #     nn.Conv2d((in_channels // 4), (out_channels // 4), 1))

        self.norm1 = LayerNorm(in_channels, eps=1e-6, data_format='channels_first')
        self.norm2 = LayerNorm(out_channels, eps=1e-6, data_format='channels_first')

    def channel_shuffle(self, x, groups):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        # reshape
        # b, c, h, w =======>  b, g, c_per, h, w
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batch_size, -1, height, width)
        return x

    def forward(self, x, gt, boundary):
        x = self.norm1(x)
        x1 = x * torch.sigmoid(gt)
        x2 = x * torch.sigmoid(boundary)
        x = x + x1 + x2
        chunks = torch.chunk(x, chunks=4, dim=1)
        group1 = chunks[0]
        group2 = chunks[1]
        group3 = chunks[2]
        group4 = chunks[3]

        # =============share_memory1_xy================
        group1 = self.conv1x1_1(group1)
        memory1 = self.share_memory1
        sharememory1 = F.interpolate(memory1, size=group1.shape[2:4], mode='bilinear', align_corners=True)
        group1 = group1 * F.gelu(self.xy_conv1(sharememory1))
        # =============share_memory2_zx================
        group2 = self.conv1x1_2(group2)
        group2 = group2.permute(0, 3, 1, 2)
        memory2 = self.share_memory2
        sharememory2 = F.interpolate(memory2, size=group2.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)
        group2 = group2 * F.gelu(self.zx_conv2(sharememory2)).unsqueeze(0)
        group2 = group2.permute(0, 2, 3, 1)
        # =============share_memory3_zy================
        group3 = self.conv1x1_3(group3)
        group3 = group3.permute(0, 2, 1, 3)
        memory3 = self.share_memory3
        sharememory3 = F.interpolate(memory3, size=group3.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)
        group3 = group3 * F.gelu(self.zy_conv3(sharememory3)).unsqueeze(0)
        group3 = group3.permute(0, 2, 1, 3)
        # =============share_memory4================
        group4 = self.conv1x1_4(group4)

        x = torch.cat((group1, group2, group3, group4), dim=1)
        x = self.norm2(x)
        x = self.channel_shuffle(x, 4)

        return x


class Upsample_block(nn.Module):
    SeedSed(seed=10)

    def __init__(self, in_channels, out_channels, boundary=True):
        super(Upsample_block, self).__init__()

        self.boundary = boundary

        self.conv1 = Light_Up(in_channels, out_channels)

        self.gt_supervised = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.boundary_supervised = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x, y):
        B, C, H, W = y.shape
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        x = x + y

        if self.boundary:
            # gt_pre,boundary_pre
            gt_pre1 = self.gt_supervised(x)
            gt_pre = torch.sigmoid(F.interpolate(gt_pre1, size=(256, 256), mode='bilinear', align_corners=False))
            boundary_pre1 = self.boundary_supervised(x)
            boundary_pre = torch.sigmoid(
                F.interpolate(boundary_pre1, size=(256, 256), mode='bilinear', align_corners=False))
            x = self.conv1(x, gt_pre1, boundary_pre1)
            return x, gt_pre, boundary_pre
        else:
            # gt_pre,boundary_pre
            gt_pre = self.gt_supervised(x)
            gt_pre = torch.sigmoid(F.interpolate(gt_pre, size=(256, 256), mode='bilinear', align_corners=False))
            x = self.conv1(x)
            return x, gt_pre

class MSFF(nn.Module):
    SeedSed(seed=10)

    def __init__(self, out_channels):
        super(MSFF, self).__init__()

        self.conv1x1_1 = nn.Conv2d(32, 8, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(24, 4, kernel_size=1)
        self.conv1x1_3 = nn.Conv2d(16, 4, kernel_size=1)
        self.conv1x1_4 = nn.Conv2d(16, 2, kernel_size=1)
        self.conv1x1_5 = nn.Conv2d(8, 2, kernel_size=1)
        self.conv1x1_6 = nn.Conv2d(8, 2, kernel_size=1)

        self.sdpc=SDPC(in_channels=3, out_channels=3)
        self.sdpc_1 = nn.Conv2d(24, 8, 1)
        self.sdpc_2 = nn.Conv2d(16, 4, 1)
        self.sdpc_3 = nn.Conv2d(8, 4, 1)
        self.sdpc_4 = nn.Conv2d(19, 2, 1)
        self.sdpc_5 = nn.Conv2d(4, 2, 1)
        self.sdpc_6 = nn.Conv2d(4, 2, 1)

        self.conv1x1_out = nn.Conv2d(4, out_channels, kernel_size=1)

        self.ca1 = saca(32)
        self.ca2 = saca(24)
        self.ca3 = saca(16)
        self.ca4 = saca(19)

    def forward(self, x6, x7, x8, x9,x_start):

        x_start=self.sdpc(x_start)
        x9=torch.cat((x9,x_start),dim=1)

        x6 = self.ca1(x6)
        x7 = self.ca2(x7)
        x8 = self.ca3(x8)
        x9 = self.ca4(x9)

        B6, C6, H6, W6 = x6.shape
        B7, C7, H7, W7 = x7.shape
        B8, C8, H8, W8 = x8.shape
        B9, C9, H9, W9 = x9.shape

        # layer1
        x6 = F.interpolate(x6, size=(H7, W7), mode='bilinear', align_corners=False)
        x6 = self.conv1x1_1(x6)
        x7_1 = self.sdpc_1(x7)
        x7_1 = torch.cat((x7_1, x6), dim=1)
        # layer2
        x7 = F.interpolate(x7, size=(H8, W8), mode='bilinear', align_corners=False)
        x7 = self.conv1x1_2(x7)
        x8_1 = self.sdpc_2(x8)
        x8_1 = torch.cat((x8_1, x7), dim=1)
        x8_2 = self.sdpc_3(x8_1)
        x7_1 = F.interpolate(x7_1, size=(H8, W8), mode='bilinear', align_corners=False)
        x7_1 = self.conv1x1_3(x7_1)
        x8_2 = torch.cat((x8_2, x7_1), dim=1)
        # layer3
        x9_1 = self.sdpc_4(x9)
        x8 = F.interpolate(x8, size=(H9, W9), mode='bilinear', align_corners=False)
        x8 = self.conv1x1_4(x8)
        x9_1 = torch.cat((x9_1, x8), dim=1)
        x9_2 = self.sdpc_5(x9_1)
        x8_1 = F.interpolate(x8_1, size=(H9, W9), mode='bilinear', align_corners=False)
        x8_1 = self.conv1x1_5(x8_1)
        x9_2 = torch.cat((x9_2, x8_1), dim=1)
        x9_3 = self.sdpc_6(x9_2)
        x8_2 = F.interpolate(x8_2, size=(H9, W9), mode='bilinear', align_corners=False)
        x8_2 = self.conv1x1_6(x8_2)
        x9_3 = torch.cat((x9_3, x8_2), dim=1)

        x = self.conv1x1_out(x9_3)

        return x

class UltraLight(nn.Module):
    SeedSed(seed=10)

    def __init__(self):
        in_chan = 3
        out_chan = 1
        super(UltraLight, self).__init__()

        self.down1 = Downsample_block(in_chan, 16)
        self.down2 = Downsample_block(16, 24)
        self.down3 = Downsample_block(24, 32)
        self.down4 = Downsample_block(32, 40)
        self.bottle1 = Downsample_block(40, 48, self_is=False)
        self.bottle2 = Downsample_block(48, 40, self_is=False)
        self.up4 = Upsample_block(40, 32)
        self.up3 = Upsample_block(32, 24)
        self.up2 = Upsample_block(24, 16)
        self.up1 = Upsample_block(16, 16)
        # self.outconv = nn.Conv2d(16, out_chan, 1)
        self.msf = MSFF(1)

    def forward(self, x):
        x_start=x
        x1, y1 = self.down1(x)
        x2, y2 = self.down2(x1)
        x3, y3 = self.down3(x2)
        x4, y4 = self.down4(y3)
        x5 = self.bottle1(x4)
        x5 = self.bottle2(x5)
        x6, gt_4, Bp_4 = self.up4(x5, y4)
        x7, gt_3, Bp_3 = self.up3(x6, y3)
        x8, gt_2, Bp_2 = self.up2(x7, y2)
        x9, gt_1, Bp_1 = self.up1(x8, y1)
        out = torch.sigmoid(self.msf(x6, x7, x8, x9,x_start))

        return (gt_1, gt_2, gt_3, gt_4), (Bp_1, Bp_2, Bp_3, Bp_4), out


if __name__ == '__main__':
    SeedSed(seed=10)
    input = torch.randn((1, 3, 256, 256)).to(device)
    model = UltraLight().to(device)
    out = model(input)
    print(out.shape)
    print(out)
