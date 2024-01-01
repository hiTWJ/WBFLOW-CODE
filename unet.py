# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import cv2
import matplotlib.pyplot as plt


# --- gaussian initialize ---
def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = weight_norm(nn.Linear(indim, outdim, bias=False), name='weight', dim=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        L_norm = torch.norm(self.L.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.L.weight.data)
        self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized)
        scores = 10 * cos_dist
        return scores


# --- flatten tensor ---
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)



# --- Conv2d module ---
class Conv2d_fw(nn.Conv2d):  # used in MAML to forward input with fast weight
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=1,
                                        bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None
        self.padding = padding


    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        return out

class Convtrans2d_fw(nn.ConvTranspose2d):  # used in MAML to forward input with fast weight
    def __init__(self, in_channels, out_channels, kernel_size, stride=2,  bias=True):
        super(Convtrans2d_fw, self).__init__(in_channels, out_channels, kernel_size=2, stride=stride,
                                        bias=bias)
        self.weight.fast = None
        # self.transposed = trasposed
        if not self.bias is None:
            self.bias.fast = None
        # self.padding = padding


    def forward(self, x):

        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv_transpose2d(x,self.weight.fast, None,stride=self.stride)
                # out = F.conv2d(x, self.weight.fast, None, stride=self.stride, padding=self.padding)
            else:
                out = super(Convtrans2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv_transpose2d(x, self.weight.fast, self.bias.fast, stride=self.stride)
            else:
                out = super(Convtrans2d_fw, self).forward(x)

        return out


# --- softplus module ---
def softplus(x):
    return torch.nn.functional.softplus(x, beta=100)

class FeatureWiseTransformation2d_fw(nn.BatchNorm2d):
    feature_augment = True

    def __init__(self, num_features, momentum=0.1, track_running_stats=True):
        super(FeatureWiseTransformation2d_fw, self).__init__(num_features, momentum=momentum,
                                                             track_running_stats=track_running_stats)

        if self.feature_augment:  # initialize {gamma, beta} with {0.3, 0.5}
            self.gamma = torch.nn.Parameter(torch.ones(1, num_features, 1, 1) * 0.3)
            self.beta = torch.nn.Parameter(torch.ones(1, num_features, 1, 1) * 0.5)
        # self.reset_parameters()
        self.BN = nn.BatchNorm2d(num_features)

    # def reset_running_stats(self):
    #     if self.track_running_stats:
    #         self.running_mean.zero_()
    #         self.running_var.fill_(1)

    def forward(self, x, step=0):
        out = self.BN(x)
        # apply feature-wise transformation
        ## self.training?
        if self.feature_augment and self.training:
            gamma = (1 + torch.randn(1, self.num_features, 1, 1, dtype=self.gamma.dtype,
                                     device=self.gamma.device) * softplus(self.gamma)).expand_as(out)
            beta = (torch.randn(1, self.num_features, 1, 1, dtype=self.beta.dtype, device=self.beta.device) * softplus(
                self.beta)).expand_as(out)
            out = gamma * x + beta

        return out


# --- feature-wise transformation layer ---
class FeatureWiseTransformation2d_fw(nn.BatchNorm2d):
    feature_augment = True

    def __init__(self, num_features, momentum=0.1, track_running_stats=True):
        super(FeatureWiseTransformation2d_fw, self).__init__(num_features, momentum=momentum,
                                                             track_running_stats=track_running_stats)

        if self.feature_augment:  # initialize {gamma, beta} with {0.3, 0.5}
            self.gamma = torch.nn.Parameter(torch.ones(1, num_features, 1, 1) * 0.3)
            self.beta = torch.nn.Parameter(torch.ones(1, num_features, 1, 1) * 0.5)
        # self.reset_parameters()
        self.BN = nn.BatchNorm2d(num_features)

    # def reset_running_stats(self):
    #     if self.track_running_stats:
    #         self.running_mean.zero_()
    #         self.running_var.fill_(1)

    def forward(self, x, step=0):
        out = self.BN(x)
        # apply feature-wise transformation
        ## self.training?
        if self.feature_augment and self.training:
            gamma = (1 + torch.randn(1, self.num_features, 1, 1, dtype=self.gamma.dtype,
                                     device=self.gamma.device) * softplus(self.gamma)).expand_as(out)
            beta = (torch.randn(1, self.num_features, 1, 1, dtype=self.beta.dtype, device=self.beta.device) * softplus(
                self.beta)).expand_as(out)
            out = gamma * x + beta

        return out

class FeatureWiseTransformation2d_fw_camera1(nn.Module):
    def __init__(self, num_features,camera_dim):
        super(FeatureWiseTransformation2d_fw_camera, self).__init__()
        "输入: num_features * 1 *1, 输出: num_features * 1 *1"
        self.conv_camera = nn.Sequential(
            nn.Conv2d(camera_dim,num_features,1),
            nn.ReLU(),
            nn.Conv2d(num_features, num_features // 4, 1),
            nn.ReLU(),
            nn.Conv2d(num_features//4,num_features,1)
        )

        "输入: num_features *3* 1, 输出: num_features*3 * 1"
        self.conv_distribution = nn.Sequential(
            nn.Conv2d(num_features,num_features//4,1),
            nn.ReLU(),
            nn.Conv2d(num_features // 4, num_features, 1)
        )

        "输入: num_features *H*W, 输出: 3*H*W"
        self.conv_feat = nn.Sequential(
            nn.Conv2d(num_features,3,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(3,3,3,padding=1)
        )



    def forward(self, x, camera):
        ## camera:偏差
        camera_ = self.conv_camera(camera)
        ## distribution：权重
        mean_ = torch.mean(x,dim=(2,3),keepdim=True)
        var_ = torch.var(x,dim=(2,3),keepdim=True)
        ## np.mean(abs(h - h.mean())**3)
        ske_ = torch.mean((x-mean_)**3,dim=(2,3),keepdim=True)
        all_ = torch.cat((mean_,var_,ske_),dim=2)
        dis_ = self.conv_distribution(all_).squeeze(3)
        ## 输入特征
        b,c,h,w = x.shape
        feat=  self.conv_feat(x).reshape((b,3,-1))
        out = torch.matmul(dis_,feat).reshape((b,c,h,w))
        out = out + camera_.expand_as(out)
        return out,camera_


class FeatureWiseTransformation2d_fw_camera(nn.Module):
    def __init__(self, num_features,camera_dim):
        super().__init__()
        "输入: num_features * 1 *1, 输出: num_features * 1 *1"
        self.conv_camera = nn.Sequential(
            nn.Conv2d(camera_dim,num_features,1),
            nn.ReLU(),
            nn.Conv2d(num_features, num_features // 4, 1),
            nn.ReLU(),
            nn.Conv2d(num_features//4,num_features,1)
        )

        "输入: num_features *3* 1, 输出: num_features*3 * 1"
        self.conv_distribution = nn.Sequential(
            nn.Conv2d(num_features,num_features//4,1),
            nn.ReLU(),
            nn.Conv2d(num_features // 4, num_features, 1)
        )

        "输入: num_features *H*W, 输出: 3*H*W"
        self.conv_feat = nn.Sequential(
            nn.Conv2d(num_features,3,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(3,3,3,padding=1)
        )
        self.sigmoid = nn.Sigmoid()



    def forward(self, x, camera):
        ## camera:偏差
        camera_ = self.conv_camera(camera)
        ## distribution：权重
        mean_ = torch.mean(x,dim=(2,3),keepdim=True)
        var_ = torch.var(x,dim=(2,3),keepdim=True)
        ## np.mean(abs(h - h.mean())**3)
        ske_ = torch.mean((x-mean_)**3,dim=(2,3),keepdim=True)
        all_ = mean_+var_+ske_
        dis_ = self.sigmoid(self.conv_distribution(all_))
        out = dis_ * x + camera_
        ## 输入特征
        # b,c,h,w = x.shape
        # feat=  self.conv_feat(x).reshape((b,3,-1))
        # out = torch.matmul(dis_,feat).reshape((b,c,h,w))
        # out = out + camera_.expand_as(out)
        return out,camera_



# --- Simple Conv Block ---
"替换backbone中的conv"
# class DoubleConvBlock(nn.Module):
#     """double conv layers block"""
#     def __init__(self, in_channels, out_channels, camera_dim1,camera_dim2):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels,out_channels,3,padding=1)
#         self.ft1 = FeatureWiseTransformation2d_fw_camera(out_channels,camera_dim1)
#         self.conv2 = nn.Conv2d(out_channels,out_channels,3,padding=1)
#         self.ft2 = FeatureWiseTransformation2d_fw_camera(out_channels,camera_dim2)
#         self.relu = nn.ReLU()
#
#
#     def forward(self, x,camera):
#         x1 = self.conv1(x)
#         x1_,camera1_ = self.ft1(x1,camera)
#         x1_ = self.relu(x1_)
#         x2 = self.conv2(x1_)
#         x2_,camera2_ = self.ft2(x2,camera1_)
#         x2_ = self.relu(x2_)
#         return x2_,camera2_






"--------U-NET--------"


class deepWBnet(nn.Module):
    def __init__(self):
        super(deepWBnet, self).__init__()
        self.n_channels = 3
        self.encoder_inc = DoubleConvBlock(self.n_channels, 24)
        self.encoder_down1 = DownBlock(24, 48)
        self.encoder_down2 = DownBlock(48, 96)
        self.encoder_down3 = DownBlock(96, 192)
        self.encoder_bridge_down = BridgeDown(192, 384)
        self.decoder_bridge_up = BridgeUP(384, 192)
        self.decoder_up1 = UpBlock(192, 96)
        self.decoder_up2 = UpBlock(96, 48)
        self.decoder_up3 = UpBlock(48, 24)
        self.decoder_out = OutputBlock(24, self.n_channels)
        # self.correction = AdaIN_wb(384)


    def forward(self, x):
        x1 = self.encoder_inc(x)
        x2 = self.encoder_down1(x1)
        x3 = self.encoder_down2(x2)
        x4 = self.encoder_down3(x3)
        x5 = self.encoder_bridge_down(x4)
        # x5 = self.correction(x5)
        x = self.decoder_bridge_up(x5)
        x = self.decoder_up1(x, x4)
        x = self.decoder_up2(x, x3)
        x = self.decoder_up3(x, x2)
        out = self.decoder_out(x, x1)
        return out


class AdaIN_wb(nn.Module):
    def __init__(self,num_features):
        super().__init__()

        self.conv_mean = nn.Sequential(
            nn.Conv2d(num_features, num_features // 4, 1),
            nn.ReLU(),
            nn.Conv2d(num_features // 4, num_features, 1)
        )

        self.conv_std = nn.Sequential(
            nn.Conv2d(num_features, num_features // 4, 1),
            nn.ReLU(),
            nn.Conv2d(num_features // 4, num_features, 1)
        )

        "输入: num_features *H*W, 输出: 3*H*W"
        # self.conv_feat = nn.Sequential(
        #     nn.Conv2d(num_features, num_features//4, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(num_features//4, num_features, 3, padding=1)
        # )
        ""
        self.sigmoid = nn.Sigmoid()

    def forward(self, content):
        size = content.size()
        content_mean, content_std = self.calc_mean_std(content)
        "feature"
        normalized_feat = (content - content_mean.expand(size)) / content_std.expand(size)
        # normalized_feat = self.conv_feat(normalized_feat)
        "mean"
        learned_mean = self.sigmoid(self.conv_mean(content_mean))
        learned_std = self.sigmoid(self.conv_std(content_std))
        # learned_mean = self.conv_mean(content_mean)
        # learned_std = self.conv_std(content_std)

        out = normalized_feat * learned_std.expand(size) + learned_mean.expand(size)
        return out

    def calc_mean_std(self,feat, eps=1e-5):
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std


class DownBlock(nn.Module):
    """Downscale block: maxpool -> double conv block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.doubelconv = DoubleConvBlock(in_channels,out_channels)


    def forward(self, x):
        x = self.maxpool(x)
        x_out = self.doubelconv(x)
        return x_out


class BridgeDown(nn.Module):
    """Downscale bottleneck block: maxpool -> conv"""
    def __init__(self, in_channels, out_channels, padding=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=False)
        )


    def forward(self, x):
        x_out = self.maxpool_conv(x)
        return x_out


class BridgeUP(nn.Module):
    """Downscale bottleneck block: conv -> transpose conv"""
    def __init__(self, in_channels, out_channels,padding = 1):
        super().__init__()
        self.conv_up = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        # x1 = self.conv1(x)
        # x2 = self.conv2(x1)
        return self.conv_up(x)


class UpBlock(nn.Module):
    """Upscale block: double conv block -> transpose conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConvBlock(in_channels * 2, in_channels)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return torch.relu(self.up(x))



class DoubleConvBlock(nn.Module):
    """double conv layers block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class OutputBlock(nn.Module):
    """Output block: double conv block -> output conv"""
    def __init__(self, in_channels, out_channels, padding=1,camera_num=15):
        super().__init__()

        self.out_conv = nn.Sequential(
            DoubleConvBlock(in_channels * 2, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )



    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        out = self.out_conv(x)
        return out












# if __name__ == '__main__':
#     #### 串行结构
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     img = cv2.imread('/home/lcx/deep_final/deep_final/example_images/01.jpg')
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (128, 128))
#     plt.imshow(img)
#     plt.show()
#     x = torch.from_numpy(img) / 255
#     x = x.unsqueeze(0)
#     x = x.permute(0, 3, 1, 2).to(dtype=torch.float32)
#     camera = torch.randn((1,1,1,1))
#
#
#     conv = unet_()
#     for n, p in conv.named_parameters():
#         print(n)
#
#     y,camera_ = conv(x,camera)
#     # x = x.to(device, dtype=torch.float32)
#
#     # x = torch.randn((1, 3, 128, 128)).to(dtype=torch.float32)
#     # x = torch.randn((2,24,128,128)).to(device='cuda', dtype=torch.float32)
#     # x_ = x.detach().cpu().numpy()
#     # plt.imshow(x_[0,0,:,:])
#     # plt.axis('off')
#     # plt.show()
#
#
#     #
#     # y_att = y_att.permute(0,2,3,1)
#     # x_ = y_att.detach().cpu().numpy()
#     # plt.imshow(x_[0,:,:,0])
#     # plt.axis('off')
#     # plt.show()

