"""
 Main blocks of the network
 Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 If you use this code, please cite the following paper:
 Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
"""
__author__ = "Mahmoud Afifi"
__credits__ = ["Mahmoud Afifi"]

import torch
import torch.nn as nn

# from arch.squeezenet.Fire import Fire
from typing import Union
import torch

import numpy as np

import math

class DoubleConvBlock(nn.Module):
    """double conv layers block"""
    def __init__(self, in_channels, out_channels, B=False):
        super().__init__()
        if B:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    """Downscale block: maxpool -> double conv block"""
    def __init__(self, in_channels, out_channels,B):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvBlock(in_channels, out_channels,B=B)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class BridgeDown(nn.Module):
    """Downscale bottleneck block: maxpool -> conv"""
    def __init__(self, in_channels, out_channels,B):
        super().__init__()
        if B:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels)
            )

        else:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.maxpool_conv(x)


class BridgeUP(nn.Module):
    """Downscale bottleneck block: conv -> transpose conv"""
    def __init__(self, in_channels, out_channels,B):
        super().__init__()
        if B:
            self.conv_up = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(in_channels),
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            )

        else:
            self.conv_up = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            )

    def forward(self, x):
        return self.conv_up(x)


class BridgeUP1(nn.Module):
    """Downscale bottleneck block: conv -> transpose conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_up = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2,output_padding=1)
        )

    def forward(self, x):
        return self.conv_up(x)



class UpBlock(nn.Module):
    """Upscale block: double conv block -> transpose conv"""
    def __init__(self, in_channels, out_channels,B):
        super().__init__()
        self.conv = DoubleConvBlock(in_channels * 2, in_channels,B)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)



    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return torch.relu(self.up(x))


class UpBlock1(nn.Module):
    """Upscale block: double conv block -> transpose conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConvBlock(in_channels * 2, in_channels)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2,output_padding=1)


    def forward(self, x1,x2):
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return torch.relu(self.up(x))


class UpBlock2(nn.Module):
    """Upscale block: double conv block -> transpose conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConvBlock(in_channels * 2, in_channels)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2,output_padding=1)


    def forward(self, x1,x2):
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return torch.relu(self.up(x))



class OutputBlock(nn.Module):
    """Output block: double conv block -> output conv"""
    def __init__(self, in_channels, out_channels,B):
        super().__init__()
        self.out_conv = nn.Sequential(
            #### 尝试加入
            DoubleConvBlock(in_channels * 2, in_channels,B),
            nn.Conv2d(in_channels, out_channels, kernel_size=1))

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        return self.out_conv(x)

class OutputBlock1(nn.Module):
    """Output block: double conv block -> output conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_conv = nn.Sequential(
            #### 尝试加入
            DoubleConvBlock(in_channels, 12),
            nn.Conv2d(12, out_channels, kernel_size=1))

    def forward(self, x):
        return self.out_conv(x)


class upsample(nn.Module):
    """Output block: double conv block -> output conv"""
    def __init__(self,):
        super().__init__()

        # squeezenet = SqueezeNetLoader(squeezenet_version).load(pretrained=True)
        self.up = nn.Sequential(
            Fire(512,32,256,256),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2),
            Fire(256, 32, 128, 128),  ## output 256
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            Fire(128, 16, 64, 64),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            Fire(64, 8, 32, 32),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            Fire(32, 8, 16, 16),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.Conv2d(16,3,kernel_size=1)
        )


    def forward(self, x):
        return self.up(x)


class FC4(torch.nn.Module):

    def __init__(self, squeezenet_version: float = 1.1):
        super().__init__()

        # SqueezeNet backbone (conv1-fire8) for extracting semantic features
        squeezenet = SqueezeNetLoader(squeezenet_version).load(pretrained=True)
        self.backbone = nn.Sequential(*list(squeezenet.children())[0][:12])

        # Final convolutional layers (conv6 and conv7) to extract semi-dense feature maps
        # self.final_convs = nn.Sequential(
        #     nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True),
        #     nn.Conv2d(512, 64, kernel_size=6, stride=1, padding=3),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5),
        #     nn.Conv2d(64, 4 , kernel_size=1, stride=1),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, x: torch.Tensor) -> Union[tuple, torch.Tensor]:
        """
        Estimate an RGB colour for the illuminant of the input image
        @param x: the image for which the colour of the illuminant has to be estimated
        @return: the colour estimate as a Tensor. If confidence-weighted pooling is used, the per-path colour estimates
        and the confidence weights are returned as well (used for visualizations)
        """
        #### self.backbone的输出尺寸为 w/16 * w/16 * 256
        out = self.backbone(x)

        # out = self.final_convs(self.backbone(x))
        return out


class Fire(nn.Module):

    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([self.expand1x1_activation(self.expand1x1(x)),
                          self.expand3x3_activation(self.expand3x3(x))], 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 通道部分做max
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # out = avg_out + max_out
        out = max_out
        return self.sigmoid(out)


class ChannelAttention_pre(nn.Module):
    def __init__(self, in_planes, ratio=3):
        super(ChannelAttention_pre, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x1,x2):
        x = torch.cat((x1,x2),1)
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        w = self.sigmoid(out)
        out = w*x
        out1 = out[:,:3,:,:]
        out2 = out[:,3:,:,:]
        return out1, out2

def get_freq_indices(method):
    # assert method is 'low'
    # num_freq = idx
    # num_freq = int(method[3:])
    if 'low' in method:  ## x 20. y 20
        num_freq = int(method[3:])
        all_low_indices_x = [0, 0, 1, 2, 1, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5]
        all_low_indices_y = [0, 1, 0, 0, 1, 2, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
        mapper_x = all_low_indices_x[:num_freq+1]
        mapper_y = all_low_indices_y[:num_freq+1]
    elif '0' in method:
        mapper_x =[0]
        mapper_y =[0]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class high_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(high_block, self).__init__()

        self.out_conv = nn.Sequential(
            DoubleConvBlock(in_channels, in_channels*2),
            nn.Conv2d(in_channels*2, out_channels, kernel_size=1))

    def forward(self,x):
        return self.out_conv(x)


class out_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(out_block, self).__init__()

        self.out_conv = nn.Sequential(
            DoubleConvBlock(in_channels, in_channels*2),
            nn.Conv2d(in_channels*2, out_channels, kernel_size=1))

    def forward(self,x1,x2):
        x = x1+x2
        return self.out_conv(x)


class out_block1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(out_block1, self).__init__()

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1))

    def forward(self,x1,x2):
        x = x1+x2
        return self.out_conv(x)

class out_block2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(out_block2, self).__init__()

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1))

    def forward(self,x1,x2):
        x = x1+x2
        return self.out_conv(x)


class high_block1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(high_block1, self).__init__()

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels, 12, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            # nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

    def forward(self,x):
        return self.out_conv(x)

class com_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(com_block, self).__init__()

        self.out_conv = nn.Sequential(
            DoubleConvBlock(in_channels, 12),
            nn.Conv2d(12, out_channels, kernel_size=1)
            # nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.out_conv(x)


class pre_img_block(nn.Module):
    def __init__(self, N, freq_sel_method='low0'):
        super(pre_img_block, self).__init__()
        self.N = N

        #### mask
        self.register_buffer('weight', self.init_cos())


        # self.weight1 =  self.weight.permute(0, 1, 3, 2)
        #
        # self.fuse_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        #
        #
        # self.fuse_weight_1.data.fill_(0.25)
        # self.fuse_weight_2.data.fill_(0.25)
        # self.fuse_weight_3.data.fill_(0.25)
        self.method = freq_sel_method

        # self.weight = self.init_cos()
        mapper_x,mapper_y = get_freq_indices(self.method)
        self.mapper_x = [temp_x * (self.N // 8) for temp_x in mapper_x]
        self.mapper_y = [temp_y * (self.N // 8) for temp_y in mapper_y]
        # print('d')

    def forward(self,x):
        ### img [b,c,h,w]
        n, c, h, w = x.shape
        ### 转成8*8的小块
        x_1 = torch.split(x, self.N, dim=2)  # [b,c,h,w] → h/8 * [b,c,8,w]
        x_2 = torch.cat(x_1, dim=1)  # → [b, c * h/8, 8, w]
        x_3 = torch.split(x_2, self.N, dim=3)  # [b, c * h/8, 8, w] → w/8 * [b, c * h/8, 8, 8]
        x_4 = torch.cat(x_3, dim=1)  # [b, c []* h/8 * w/8 , 8, 8]
        ### DCT

        fre = torch.einsum('pijk,qtkl->qtjl', self.weight, x_4)  # (256,8,8)
        fre = torch.einsum('pijk,qtkl->pijl', fre, self.weight.permute(0, 1, 3, 2)) # (256,8,8)

        ### index
        fre_low = self.low_fre(self.mapper_x, self.mapper_y, fre)
        # fre_low.cuda()
        # fre = zigzag(fre)
        # fre.unsqueeze(dim=3)
        # fre[:,:,(self.T+1):] = 0
        # fre_low = inverse_zigzag(fre,self.N,self.N)

        ### x_low

        x_low = torch.einsum('pijk,qtkl->qtjl', self.weight.permute(0, 1, 3, 2), fre_low)
        x_low = torch.einsum('pijk,qtkl->pijl', x_low, self.weight)
        n_low, c_low, h_low, w_low = x_low.shape
        x_low = torch.split(x_low, int(c_low/(w/self.N)), dim=1)
        x_low = torch.cat(x_low, dim=3)
        x_low = torch.split(x_low, c, dim=1)
        x_low = torch.cat(x_low, dim=2)
        # print(x[0, 0, :8, :8])
        # print(x_low[0,0,:8,:8])

        ### x_high
        x_high = x-x_low

        return x_low, x_high



    def low_fre(self,mapper_x, mapper_y,fre):
        n, c, h, w = fre.shape
        low = torch.zeros_like(fre)
        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            low[:,:,u_x,v_y] = fre[:,:,u_x,v_y]
        # print(low)
        return low


    def init_cos(self):
        A = torch.zeros((1, 1, self.N, self.N))
        A[0, 0, 0, :] = 1 * math.sqrt(1 / self.N)
        for i in range(1, self.N):
            for j in range(self.N):
                A[0, 0, i, j] = math.cos(math.pi * i * (2 * j + 1) / (2 * self.N)
                                    ) * math.sqrt(2 / self.N)
        return A


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,
                                                                                                     height,
                                                                                                     height).permute(0,
                                                                                                                     2,
                                                                                                                     1,
                                                                                                                     3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W) + x

class pre_img_block1(nn.Module):
    def __init__(self, N, freq_sel_method='low0'):
        super(pre_img_block1, self).__init__()
        self.N = N

        #### mask
        self.register_buffer('weight', self.init_cos())


        # self.weight1 =  self.weight.permute(0, 1, 3, 2)
        #
        # self.fuse_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        #
        #
        # self.fuse_weight_1.data.fill_(0.25)
        # self.fuse_weight_2.data.fill_(0.25)
        # self.fuse_weight_3.data.fill_(0.25)
        self.method = freq_sel_method

        # self.weight = self.init_cos()
        mapper_x,mapper_y = self.get_freq_indices2(self.method)
        self.mapper_x = [temp_x * (self.N // 8) for temp_x in mapper_x]
        self.mapper_y = [temp_y * (self.N // 8) for temp_y in mapper_y]
        # print('d')

    def forward(self,x):
        ### img [b,c,h,w]
        n, c, h, w = x.shape
        ### 转成8*8的小块
        x_1 = torch.split(x, self.N, dim=2)  # [b,c,h,w] → h/8 * [b,c,8,w]
        x_2 = torch.cat(x_1, dim=1)  # → [b, c * h/8, 8, w]
        x_3 = torch.split(x_2, self.N, dim=3)  # [b, c * h/8, 8, w] → w/8 * [b, c * h/8, 8, 8]
        x_4 = torch.cat(x_3, dim=1)  # [b, c []* h/8 * w/8 , 8, 8]
        ### DCT

        fre = torch.einsum('pijk,qtkl->qtjl', self.weight, x_4)  # (256,8,8)
        fre = torch.einsum('pijk,qtkl->pijl', fre, self.weight.permute(0, 1, 3, 2)) # (256,8,8)

        ### index
        fre_low = self.low_fre(self.mapper_x, self.mapper_y, fre)
        # fre_low.cuda()
        # fre = zigzag(fre)
        # fre.unsqueeze(dim=3)
        # fre[:,:,(self.T+1):] = 0
        # fre_low = inverse_zigzag(fre,self.N,self.N)

        ### x_low

        x_low = torch.einsum('pijk,qtkl->qtjl', self.weight.permute(0, 1, 3, 2), fre_low)
        x_low = torch.einsum('pijk,qtkl->pijl', x_low, self.weight)
        n_low, c_low, h_low, w_low = x_low.shape
        x_low = torch.split(x_low, int(c_low/(w/self.N)), dim=1)
        x_low = torch.cat(x_low, dim=3)
        x_low = torch.split(x_low, c, dim=1)
        x_low = torch.cat(x_low, dim=2)
        # print(x[0, 0, :8, :8])
        # print(x_low[0,0,:8,:8])

        ### x_high
        x_high = x-x_low

        return x_low, x_high

    def get_freq_indices2(self, method):
        # assert method is 'low'
        # num_freq = idx
        # num_freq = int(method[3:])
        if 'low' in method:  ## x 20. y 20
            num_freq = int(method[3:])
            all_low_indices_x = [0, 0, 1, 2, 1, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5]
            all_low_indices_y = [0, 1, 0, 0, 1, 2, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
            mapper_x = [0,all_low_indices_x[num_freq]]
            mapper_y = [0,all_low_indices_y[num_freq]]
        elif '0' in method:
            mapper_x = [0]
            mapper_y = [0]
        else:
            raise NotImplementedError
        return mapper_x, mapper_y

    def low_fre(self,mapper_x, mapper_y,fre):
        n, c, h, w = fre.shape
        low = torch.zeros_like(fre)
        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            low[:,:,u_x,v_y] = fre[:,:,u_x,v_y]
        # print(fre)
        # print(low)
        return low


    def init_cos(self):
        A = torch.zeros((1, 1, self.N, self.N))
        A[0, 0, 0, :] = 1 * math.sqrt(1 / self.N)
        for i in range(1, self.N):
            for j in range(self.N):
                A[0, 0, i, j] = math.cos(math.pi * i * (2 * j + 1) / (2 * self.N)
                                    ) * math.sqrt(2 / self.N)
        return A

class MultiChannelAttentionLayer_pre(torch.nn.Module):
    def __init__(self, channel, dct_h=8, dct_w=8, reduction=16, freq_sel_method='low2'):
        super(MultiChannelAttentionLayer_pre, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w
        # self.freq_sel_method1 = 'low9'
        # self.freq_sel_method2 = 'low5'
        # self.freq_sel_method3 = 'low20'
        # self.freq_sel_method4 = 'low20'

        # Mx, My = [],[]
        # for i in range(len(freq_sel_method)):
        #     mapper_x, mapper_y = self.map(freq_sel_method[i])

        mapper_x, mapper_y = self.get_freq_indices1(freq_sel_method)
        # self.num_split = len(mapper_x)
        self.num_split = 1
        # mapper_x = mapper_x * (dct_h // 8)
        # mapper_y = mapper_y * (dct_w // 8)
        mapper_x = [temp_x * (dct_h // 8) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 8) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)

        # mapper_x1, mapper_y1 = get_freq_indices(self.freq_sel_method1)
        # # self.num_split = len(mapper_x)
        # self.num_split = 1
        # # mapper_x = mapper_x * (dct_h // 8)
        # # mapper_y = mapper_y * (dct_w // 8)
        # mapper_x1 = [temp_x * (dct_h // 8) for temp_x in mapper_x1]
        # mapper_y1 = [temp_y * (dct_w // 8) for temp_y in mapper_y1]
        # # make the frequencies in different sizes are identical to a 7x7 frequency space
        # # eg, (2,2) in 14x14 is identical to (1,1) in 7x7
        # self.dct_layer1 = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x1, mapper_y1, channel)
        # #


        # mapper_x2, mapper_y2 = get_freq_indices(self.freq_sel_method2)
        # # self.num_split = len(mapper_x)
        # self.num_split = 1
        # # mapper_x = mapper_x * (dct_h // 8)
        # # mapper_y = mapper_y * (dct_w // 8)
        # mapper_x2 = [temp_x * (dct_h // 8) for temp_x in mapper_x2]
        # mapper_y2 = [temp_y * (dct_w // 8) for temp_y in mapper_y2]
        # self.dct_layer2 = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x2, mapper_y2, channel)

        # mapper_x3, mapper_y3 = get_freq_indices(self.freq_sel_method3)
        # # self.num_split = len(mapper_x)
        # self.num_split = 1
        # # mapper_x = mapper_x * (dct_h // 8)
        # # mapper_y = mapper_y * (dct_w // 8)
        # mapper_x3 = [temp_x * (dct_h // 8) for temp_x in mapper_x3]
        # mapper_y3 = [temp_y * (dct_w // 8) for temp_y in mapper_y3]
        # self.dct_layer3 = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x3, mapper_y3, channel)
        #
        # mapper_x4, mapper_y4 = get_freq_indices(self.freq_sel_method4)
        # # self.num_split = len(mapper_x)
        # self.num_split = 1
        # # mapper_x = mapper_x * (dct_h // 8)
        # # mapper_y = mapper_y * (dct_w // 8)
        # mapper_x4 = [temp_x * (dct_h // 8) for temp_x in mapper_x4]
        # mapper_y4 = [temp_y * (dct_w // 8) for temp_y in mapper_y4]
        # self.dct_layer4 = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x4, mapper_y4, channel)

        # self.fc = nn.Sequential(
        #     nn.Conv2d(channel, channel // reduction, 1, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(channel // reduction, channel, 1, bias=False),
        #     # nn.Sigmoid()
        # )

        self.fc1 = nn.Conv2d(channel, channel // 3, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(channel // 3, channel, 1, bias=False)


        self.sigmoid = nn.Sigmoid()



        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)


    def forward(self, x1,x2):
        x = torch.cat((x1,x2),dim=1)
        n, c, h, w = x.shape
        # x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        else:
            x_pooled = x
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        # y = self.dct_layer(x_pooled)

        # avg_out = self.fc(self.avg_pool(x)).view(n, c, 1, 1)
        # max_out = self.fc(self.max_pool(x)).view(n, c, 1, 1)
        # fre_out = self.fc(self.dct_layer(x_pooled)).view(n, c, 1, 1)

        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        fre_out = self.fc2(self.relu1(self.fc1(self.dct_layer(x_pooled))))

        w = avg_out + max_out + fre_out
        # w = avg_out + fre_out
        # w = avg_out + fre_out1 + fre_out2 + fre_out3
        w = self.sigmoid(w)
        out = w*x
        out1 = out[:,:3,:,:]
        out2 = out[:,3:,:,:]
        return out1,out2

    def get_freq_indices1(self,method):
        # assert method is 'low'
        # num_freq = idx
        # num_freq = int(method[3:])
        if 'low' in method:  ## x 20. y 20
            num_freq = int(method[3:])
            all_low_indices_x = [0, 0, 1, 2, 1, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5]
            all_low_indices_y = [0, 1, 0, 0, 1, 2, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
            mapper_x = [all_low_indices_x[num_freq]]
            mapper_y = [all_low_indices_y[num_freq]]
        elif '0' in method:
            mapper_x = [0]
            mapper_y = [0]
        else:
            raise NotImplementedError
        return mapper_x, mapper_y

class MultiChannelAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h=8, dct_w=8, reduction=16, freq_sel_method='low2'):
        super(MultiChannelAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w
        # self.freq_sel_method1 = 'low9'
        # self.freq_sel_method2 = 'low5'
        # self.freq_sel_method3 = 'low20'
        # self.freq_sel_method4 = 'low20'

        # Mx, My = [],[]
        # for i in range(len(freq_sel_method)):
        #     mapper_x, mapper_y = self.map(freq_sel_method[i])

        mapper_x, mapper_y = self.get_freq_indices1(freq_sel_method)
        # self.num_split = len(mapper_x)
        self.num_split = 1
        # mapper_x = mapper_x * (dct_h // 8)
        # mapper_y = mapper_y * (dct_w // 8)
        mapper_x = [temp_x * (dct_h // 8) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 8) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)

        # mapper_x1, mapper_y1 = get_freq_indices(self.freq_sel_method1)
        # # self.num_split = len(mapper_x)
        # self.num_split = 1
        # # mapper_x = mapper_x * (dct_h // 8)
        # # mapper_y = mapper_y * (dct_w // 8)
        # mapper_x1 = [temp_x * (dct_h // 8) for temp_x in mapper_x1]
        # mapper_y1 = [temp_y * (dct_w // 8) for temp_y in mapper_y1]
        # # make the frequencies in different sizes are identical to a 7x7 frequency space
        # # eg, (2,2) in 14x14 is identical to (1,1) in 7x7
        # self.dct_layer1 = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x1, mapper_y1, channel)
        # #


        # mapper_x2, mapper_y2 = get_freq_indices(self.freq_sel_method2)
        # # self.num_split = len(mapper_x)
        # self.num_split = 1
        # # mapper_x = mapper_x * (dct_h // 8)
        # # mapper_y = mapper_y * (dct_w // 8)
        # mapper_x2 = [temp_x * (dct_h // 8) for temp_x in mapper_x2]
        # mapper_y2 = [temp_y * (dct_w // 8) for temp_y in mapper_y2]
        # self.dct_layer2 = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x2, mapper_y2, channel)

        # mapper_x3, mapper_y3 = get_freq_indices(self.freq_sel_method3)
        # # self.num_split = len(mapper_x)
        # self.num_split = 1
        # # mapper_x = mapper_x * (dct_h // 8)
        # # mapper_y = mapper_y * (dct_w // 8)
        # mapper_x3 = [temp_x * (dct_h // 8) for temp_x in mapper_x3]
        # mapper_y3 = [temp_y * (dct_w // 8) for temp_y in mapper_y3]
        # self.dct_layer3 = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x3, mapper_y3, channel)
        #
        # mapper_x4, mapper_y4 = get_freq_indices(self.freq_sel_method4)
        # # self.num_split = len(mapper_x)
        # self.num_split = 1
        # # mapper_x = mapper_x * (dct_h // 8)
        # # mapper_y = mapper_y * (dct_w // 8)
        # mapper_x4 = [temp_x * (dct_h // 8) for temp_x in mapper_x4]
        # mapper_y4 = [temp_y * (dct_w // 8) for temp_y in mapper_y4]
        # self.dct_layer4 = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x4, mapper_y4, channel)

        # self.fc = nn.Sequential(
        #     nn.Conv2d(channel, channel // reduction, 1, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(channel // reduction, channel, 1, bias=False),
        #     # nn.Sigmoid()
        # ) nn.Linear(channel, 24, bias=False)

        self.fc1 = nn.Conv2d(channel, channel // 16,1,  bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(channel // 16, channel,1, bias=False)


        self.sigmoid = nn.Sigmoid()



        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)


    def forward(self, x):
        n, c, h, w = x.shape
        # x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        else:
            x_pooled = x
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        # y = self.dct_layer(x_pooled)

        # avg_out = self.fc(self.avg_pool(x)).view(n, c, 1, 1)
        # max_out = self.fc(self.max_pool(x)).view(n, c, 1, 1)
        # fre_out = self.fc(self.dct_layer(x_pooled)).view(n, c, 1, 1)

        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        fre_out = self.fc2(self.relu1(self.fc1(self.dct_layer(x_pooled))))
        # fre_out1 = self.fc(self.dct_layer1(x_pooled)).view(n, c, 1, 1)
        # fre_out2 = self.fc(self.dct_layer2(x_pooled)).view(n, c, 1, 1)
        # fre_out3 = self.fc(self.dct_layer3(x_pooled)).view(n, c, 1, 1)
        # fre_out4 = self.fc(self.dct_layer2(x_pooled)).view(n, c, 1, 1)
        # w = avg_out + max_out + fre_out1 + fre_out2
        # w = avg_out  + fre_out1 + fre_out2
        w = avg_out + max_out + fre_out
        # w = avg_out + fre_out
        # w = avg_out + fre_out1 + fre_out2 + fre_out3
        w = self.sigmoid(w)
        return w

    def get_freq_indices1(self,method):
        # assert method is 'low'
        # num_freq = idx
        # num_freq = int(method[3:])
        if 'low' in method:  ## x 20. y 20
            num_freq = int(method[3:])
            all_low_indices_x = [0, 0, 1, 2, 1, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5]
            all_low_indices_y = [0, 1, 0, 0, 1, 2, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
            mapper_x = [all_low_indices_x[num_freq]]
            mapper_y = [all_low_indices_y[num_freq]]
        elif '0' in method:
            mapper_x = [0]
            mapper_y = [0]
        else:
            raise NotImplementedError
        return mapper_x, mapper_y



class MultiChannelAttentionLayer1(torch.nn.Module):
    def __init__(self, channel, dct_h=8, dct_w=8, reduction=16, freq_sel_method1='low16',freq_sel_method2='low17',freq_sel_method3='low19'):
        super(MultiChannelAttentionLayer1, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        self.dct_layer1 = self.dctlayer(freq_sel_method1, dct_h, dct_w, channel)
        self.dct_layer2 = self.dctlayer(freq_sel_method2, dct_h, dct_w, channel)
        if not (freq_sel_method3 == None):
            self.dct_layer3 = self.dctlayer(freq_sel_method3, dct_h, dct_w, channel)
            self.E = True
        else:
            self.E = False

        self.fc1 = nn.Conv2d(channel, channel // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(channel // 16, channel, 1, bias=False)
        # print(freq_sel_method1)
        # print(freq_sel_method2)
        # print(freq_sel_method3)

        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def dctlayer(self,freq_sel_method1,dct_h,dct_w,channel):
        mapper_x1, mapper_y1 = self.get_freq_indices1(freq_sel_method1)
        # self.num_split = len(mapper_x)
        # num_split = 1
        # mapper_x = mapper_x * (dct_h // 8)
        # mapper_y = mapper_y * (dct_w // 8)
        mapper_x1 = [temp_x * (dct_h // 8) for temp_x in mapper_x1]
        mapper_y1 = [temp_y * (dct_w // 8) for temp_y in mapper_y1]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        dct_layer1 = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x1, mapper_y1, channel)
        return dct_layer1

    def forward(self, x):
        n, c, h, w = x.shape
        # x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        else:
            x_pooled = x
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        # y = self.dct_layer(x_pooled)
        # avg_out = self.fc(self.avg_pool(x)).view(n, c, 1, 1)
        # max_out = self.fc(self.max_pool(x)).view(n, c, 1, 1)
        # fre_out = self.fc(self.dct_layer(x_pooled)).view(n, c, 1, 1)

        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        fre_out1 = self.fc2(self.relu1(self.fc1(self.dct_layer1(x_pooled))))
        fre_out2 = self.fc2(self.relu1(self.fc1(self.dct_layer2(x_pooled))))
        if self.E:
            fre_out3 = self.fc2(self.relu1(self.fc1(self.dct_layer3(x_pooled))))
            w = avg_out + max_out + fre_out1 + fre_out2 + fre_out3
        else:
            w = avg_out + max_out + fre_out1 + fre_out2

        # w = avg_out + fre_out
        # w = avg_out + fre_out1 + fre_out2 + fre_out3
        w = self.sigmoid(w)
        return w

    def get_freq_indices1(self,method):
        # assert method is 'low'
        # num_freq = idx
        # num_freq = int(method[3:])
        if 'low' in method:  ## x 20. y 20
            num_freq = int(method[3:])
            all_low_indices_x = [0, 0, 1, 2, 1, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5]
            all_low_indices_y = [0, 1, 0, 0, 1, 2, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
            mapper_x = [all_low_indices_x[num_freq]]
            mapper_y = [all_low_indices_y[num_freq]]
        elif '0' in method:
            mapper_x = [0]
            mapper_y = [0]
        else:
            raise NotImplementedError
        return mapper_x, mapper_y




class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter1(height, width, mapper_x, mapper_y, channel))

        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape
        # x = torch.nn.functional.adaptive_avg_pool2d(x, (8, 8))
        x = x * self.weight
        result = torch.sum(x, dim=[2, 3])
        result = torch.unsqueeze(result,dim=2)
        result = torch.unsqueeze(result, dim=3)
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter1(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,
                                                                                           tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y)
                    # print('done')

        return dct_filter

class MultiSpectralAttentionLayer_fa(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction = 16, freq_sel_method = 'top16'):
        super(MultiSpectralAttentionLayer_fa, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices_fa(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)

        self.fc1 = nn.Linear(channel, 24, bias=False)
        self.re = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(24, channel, bias=False)
        self.s = nn.Sigmoid()


    def forward(self, x):
        n,c,h,w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled).view(n,c)
        # print(y.size)
        y1 = self.fc1(y)
        y2 = self.re(y1)
        y3 = self.fc2(y2)
        y4 = self.s(y3).view(n, c, 1, 1)
        # print(y4.size)
        # y = self.fc(y).view(n, c, 1, 1)
        return y4

def get_freq_indices_fa(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y