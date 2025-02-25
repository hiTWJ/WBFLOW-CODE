import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi, exp
import numpy as np
from scipy import linalg as la

logabs = lambda x: torch.log(torch.abs(x))

# --- feature-wise transformation layer ---
class FeatureWiseTransformation2d_fw(nn.BatchNorm2d):
  feature_augment = True
  def __init__(self, num_features, momentum=0.1, track_running_stats=True):
    super(FeatureWiseTransformation2d_fw, self).__init__(num_features, momentum=momentum, track_running_stats=track_running_stats)
    self.weight.fast = None
    self.bias.fast = None
    "???"
    if self.track_running_stats:
      self.register_buffer('running_mean', torch.zeros(num_features))
      self.register_buffer('running_var', torch.zeros(num_features))
    if self.feature_augment: # initialize {gamma, beta} with {0.3, 0.5}
      self.gamma = torch.nn.Parameter(torch.ones(1, num_features, 1, 1)*0.3)
      self.beta = torch.nn.Parameter(torch.ones(1, num_features, 1, 1)*0.5)
    self.reset_parameters()

  def reset_running_stats(self):
    if self.track_running_stats:
      self.running_mean.zero_()
      self.running_var.fill_(1)

  def forward(self, x, step=0):
    if self.weight.fast is not None and self.bias.fast is not None:
      weight = self.weight.fast
      bias = self.bias.fast
    else:
      weight = self.weight
      bias = self.bias
    if self.track_running_stats:
      out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training, momentum=self.momentum)
    else:
      out = F.batch_norm(x, torch.zeros_like(x), torch.ones_like(x), weight, bias, training=True, momentum=1)

    # apply feature-wise transformation
    ## self.training?
    if self.feature_augment and self.training:
      gamma = (1 + torch.randn(1, self.num_features, 1, 1, dtype=self.gamma.dtype, device=self.gamma.device)*softplus(self.gamma)).expand_as(out)
      beta = (torch.randn(1, self.num_features, 1, 1, dtype=self.beta.dtype, device=self.beta.device)*softplus(self.beta)).expand_as(out)
      out = gamma*x + beta

    return out


def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


# feature-level AdaIN
class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, content, style):
        assert (content.size()[:2] == style.size()[:2])
        size = content.size()
        style_mean, style_std = calc_mean_std(style)
        content_mean, content_std = calc_mean_std(content)
        normalized_feat = (content - content_mean.expand(size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

class AdaIN_wb(nn.Module):
    def __init__(self,num_features):
        super().__init__()

        # self.conv_mean = nn.Sequential(
        #     nn.Conv2d(num_features, num_features // 4, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(num_features // 4, num_features, 1)
        # )
        #
        # self.conv_std = nn.Sequential(
        #     nn.Conv2d(num_features, num_features // 4, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(num_features // 4, num_features, 1)
        # )


        ## 48*48
        self.conv_mean = nn.Sequential(
                InvConv2d(num_features, num_features),
                InvConv2d(num_features, num_features),
                InvConv2d(num_features, num_features)
            )

        self.conv_std = nn.Sequential(
            InvConv2d(num_features, num_features),
            InvConv2d(num_features, num_features),
            InvConv2d(num_features, num_features)
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, content):
        size = content.size()
        content_mean, content_std = calc_mean_std(content)
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


class AdaIN_wb_cam(nn.Module):
    def __init__(self, num_features, cam_num):
        super().__init__()

        ## 48*48
        self.conv_mean = nn.Sequential(
            InvConv2d(num_features, num_features),
            InvConv2d(num_features, num_features)
        )

        self.conv_std = nn.Sequential(
            InvConv2d(num_features, num_features),
            InvConv2d(num_features, num_features)
        )

        self.sigmoid = nn.Sigmoid()

        self.cam = camera_conv2d_rever(cam_num, 96)



    def forward(self, content, camidx):
        "cam-weight"
        "mean"
        tran_weight = self.cam(camidx)
        tran_weight, _ = torch.linalg.qr(tran_weight)
        trans_weight_mean, trans_weight_std = torch.split(tran_weight, [48, 48], dim=1)
        tran_weight_mean = torch.reshape(trans_weight_mean, (-1, 48, 1, 1, 1))
        b_mean = tran_weight_mean.shape[0]
        tran_weight_std = torch.reshape(trans_weight_std, (-1, 48, 1, 1, 1))
        b_std = tran_weight_std.shape[0]

        size = content.size()
        content_mean, content_std = calc_mean_std(content)
        "feature"
        normalized_feat = (content - content_mean.expand(size)) / content_std.expand(size)
        # normalized_feat = self.conv_feat(normalized_feat)
        "sta"
        learned_mean = self.conv_mean(content_mean)
        learned_std = self.conv_mean(content_std)

        "cam_sta_mean"
        learned_mean_cam = torch.zeros_like(learned_mean)
        for ii in range(b_mean):
            cam_fea = learned_mean[ii].unsqueeze(0)
            cam_wei = tran_weight_mean[ii]
            learned_mean_cam[ii] = F.conv2d(cam_fea, cam_wei, groups=48)
        learned_mean_cam = self.sigmoid(learned_mean_cam)

        "cam_sta_std"
        learned_std_cam = torch.zeros_like(learned_std)
        for ii in range(b_std):
            cam_fea = learned_std[ii].unsqueeze(0)
            cam_wei = tran_weight_std[ii]
            learned_std_cam[ii] = F.conv2d(cam_fea, cam_wei, groups=48)
        learned_std_cam = self.sigmoid(learned_std_cam)
        out = normalized_feat * learned_std_cam.expand(size) + learned_mean_cam.expand(size)
        return out



class AdaIN_wb_feat(nn.Module):
    def __init__(self,num_features):
        super().__init__()

        "通道之间的信息互通牵制"
        self.conv_feat = nn.Sequential(
            nn.Conv2d(num_features, num_features//4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_features//4, num_features//16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_features//16, num_features // 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_features // 4, num_features, 3, padding=1),
            nn.ReLU()
        )


    def forward(self, content):
        size = content.size()
        content_mean, content_std = calc_mean_std(content)
        "feature_ori"
        normalized_feat = (content - content_mean.expand(size)) / content_std.expand(size)
        # normalized_feat = self.conv_feat(normalized_feat)
        "feature_learned"
        learned_content = self.conv_feat(content)
        learned_mean, learned_std = calc_mean_std(learned_content)
        out = normalized_feat * learned_std.expand(size) + learned_mean.expand(size)
        return out




class ActNorm(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .permute(1, 0, 2, 3)
            )
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height, width = input.shape
        # sha = input.shape
        # height = sha[-2]
        # width = sha[-1]

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class InvConv2d(nn.Module):
    def __init__(self, in_channel, out_channel=None):
        super().__init__()

        if out_channel is None:
            out_channel = in_channel
        # weight = torch.randn(out_channel, in_channel)
        weight = torch.randn(in_channel, out_channel)
        q, _ = torch.linalg.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, _, height, width = input.shape
        out = F.conv2d(input, self.weight)
        return out

    def reverse(self, output):
        return F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )

class InvConv2d_c(nn.Module):
    def __init__(self, in_channel, out_channel=None):
        super().__init__()

        if out_channel is None:
            out_channel = in_channel
        # weight = torch.randn(out_channel, in_channel)
        weight = torch.randn(in_channel, out_channel)
        # q, _ = torch.linalg.qr(weight)
        weight = weight.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, _, height, width = input.shape
        out = F.conv2d(input, self.weight)
        return out

    def reverse(self, output):
        return F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )


class camera_conv2d(nn.Module):
    def __init__(self, cam_size, weight_size):
        super().__init__()
        self.conv_cam = nn.Sequential(
            nn.Conv2d(cam_size, weight_size, 1),
            nn.ReLU(),
            nn.Conv2d(weight_size, weight_size, 1),
            nn.ReLU(),
        )



    def forward(self, input):
        out = self.conv_cam(input)
        return out


class camera_conv2d_rever(nn.Module):
    def __init__(self, cam_size, weight_size):
        super().__init__()

        self.affine = False

        self.conv_cam = nn.Sequential(
            ActNorm(cam_size),
            nn.Conv2d(cam_size,cam_size*2,1),
            AffineCoupling(cam_size*2, affine=self.affine),

            ActNorm(cam_size * 2),
            nn.Conv2d(cam_size*2, weight_size,1),
            AffineCoupling(weight_size, affine=self.affine),

            ActNorm(weight_size),
            nn.Conv2d(weight_size, weight_size,1),
            AffineCoupling(weight_size, affine=self.affine)
        )

        # self.conv_cam = nn.Sequential(
        #     ActNorm(cam_size),
        #     nn.Conv2d(cam_size, cam_size * 2, 1),
        #     AffineCoupling(cam_size * 2, affine=self.affine),
        #
        #     ActNorm(cam_size * 2),
        #     nn.Conv2d(cam_size * 2, cam_size * 4, 1),
        #     AffineCoupling(cam_size * 4, affine=self.affine),
        #
        #     ActNorm(cam_size * 4),
        #     nn.Conv2d(cam_size * 4, weight_size, 1),
        #     AffineCoupling(weight_size, affine=self.affine),
        #
        #     ActNorm(weight_size),
        #     nn.Conv2d(weight_size, weight_size, 1),
        #     AffineCoupling(weight_size, affine=self.affine)
        #
        # )

        # self.conv_cam = nn.Sequential(
        #     ActNorm(cam_size),
        #     nn.Conv2d(cam_size, cam_size * 2, 1),
        #
        #     ActNorm(cam_size * 2),
        #     nn.Conv2d(cam_size * 2, weight_size, 1),
        #
        #     ActNorm(weight_size),
        #     nn.Conv2d(weight_size, weight_size, 1),
        #
        #     ActNorm(weight_size),
        #     nn.Conv2d(weight_size, weight_size, 1)
        # )

    def forward(self, input):
        out = self.conv_cam(input)
        return out



class InvConv2d_camera_trans(nn.Module):
    def __init__(self, camera_weight, in_channel, out_channel=None):
        super().__init__()
        "camera_weight: 48*48"

        if out_channel is None:
            out_channel = in_channel
        # weight = torch.randn(out_channel, in_channel)
        weight = camera_weight
        # weight = torch.randn(in_channel, out_channel)
        q, _ = torch.linalg.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, _, height, width = input.shape
        out = F.conv2d(input, self.weight)
        return out

    def reverse(self, output):
        return F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )

class InvConv2dLU(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        if out_channel is None:
            out_channel = in_channel
        weight = np.random.randn(in_channel, out_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p.copy())
        w_l = torch.from_numpy(w_l.copy())
        w_s = torch.from_numpy(w_s.copy())
        w_u = torch.from_numpy(w_u.copy())

        self.register_buffer('w_p', w_p)
        self.register_buffer('u_mask', torch.from_numpy(u_mask))
        self.register_buffer('l_mask', torch.from_numpy(l_mask))
        self.register_buffer('s_sign', torch.sign(w_s))
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, height, width = input.shape
        weight = self.calc_weight()
        out = F.conv2d(input, weight)
        return out

    def calc_weight(self):
        weight = (
                self.w_p
                @ (self.w_l * self.l_mask + self.l_eye)
                @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        weight = self.calc_weight()

        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)

        return out


class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=512, affine=True):
        super().__init__()

        self.affine = affine

        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
        )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1)
            s = torch.exp(log_s)
            s = F.sigmoid(s + 2)
            out_a = s * in_a + t
            out_b = (in_b + t) * s

        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out

        return torch.cat([in_a, out_b], 1)

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = F.sigmoid(log_s + 2)
            # in_a = (out_a - t) / s
            in_b = out_b / s - t

        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out

        return torch.cat([out_a, in_b], 1)


class Flow(nn.Module):
    def __init__(self, in_channel, out_channel=None,use_coupling=True, affine=True, conv_lu=True):
        super().__init__()

        self.actnorm = ActNorm(in_channel)

        if conv_lu:
            self.invconv = InvConv2dLU(in_channel,out_channel)
        else:
            self.invconv = InvConv2d(in_channel,out_channel)

        self.use_coupling = use_coupling
        if self.use_coupling:
            self.coupling = AffineCoupling(in_channel, affine=affine)

    def forward(self, input):
        input = self.actnorm(input)
        input = self.invconv(input)
        if self.use_coupling:
            input = self.coupling(input)
        return input

    def reverse(self, input):
        if self.use_coupling:
            input = self.coupling.reverse(input)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)

        return input


def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class Block(nn.Module):
    def __init__(self, in_channel, n_flow, affine=True, conv_lu=True):
        super().__init__()
        "n_flow=8"

        squeeze_dim = in_channel * 4


        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, affine=affine, conv_lu=conv_lu))

    def forward(self, input):
        b_size, n_channel, height, width = input.shape
        "变成block"
        squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)
        for flow in self.flows:
            out = flow(out)
        return out

    def reverse(self, output, reconstruct=False):
        input = output
        for flow in self.flows[::-1]:
            input = flow.reverse(input)

        b_size, n_channel, height, width = input.shape

        unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(
            b_size, n_channel // 4, height * 2, width * 2
        )

        return unsqueezed

class Camera_Block(nn.Module):
    def __init__(self, in_channel, out_channel, affine=True, conv_lu=False):
        super().__init__()
        "n_flow=8"
        self.flow = Flow(in_channel=in_channel, out_channel=out_channel, affine=affine, conv_lu=conv_lu)

    def forward(self, input):
        # b_size, n_channel, height, width = input.shape
        out = self.flow(input)
        return out

    def reverse(self, output, reconstruct=False):
        input = self.flow.reverse(output)
        return input

class Camera_Glow(nn.Module):
    def __init__(self, in_channel, cam_num, n_flow,n_block, affine=True, conv_lu=True):
        super().__init__()
        "n_flow=8, n_block=2"

        "image"
        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu))
            n_channel *= 4

        self.blocks.append(Block(n_channel, n_flow, affine=affine))

        "48?"
        self.cam = camera_conv2d_rever(cam_num,48)
        # self.cam = camera_conv2d(cam_num,48)
        self.wb = AdaIN_wb(num_features=48)
        # self.actnorm = ActNorm(48)


    def forward(self, input,camidx, forward=True):
        ""
        if forward:
            return self._forward(input,camidx)
        else:
            return self._reverse(input,camidx)

    def _forward(self, input,camidx):
        "camera trans weights"
        tran_weight = self.cam(camidx)
        tran_weight, _ = torch.linalg.qr(tran_weight)
        tran_weight = torch.reshape(tran_weight,(-1,48,1,1,1))
        b = tran_weight.shape[0]
        # tran_weight = tran_weight.unsqueeze(2).unsqueeze(3)
        "common mapping"
        z = input
        for block in self.blocks:
            z = block(z)

        # z = self.actnorm(z)
        z_cam = torch.zeros_like(z)
        for ii in range(b):
            cam_fea = z[ii].unsqueeze(0)
            cam_wei = tran_weight[ii]
            z_cam[ii] = F.conv2d(cam_fea,cam_wei,groups=48)

        return z_cam,tran_weight

    def _reverse(self, z_cam,tran_weight):
        ""
        "wb"
        wb_cam = self.wb(z_cam)
        "cam"
        b = tran_weight.shape[0]
        wb_out = torch.zeros_like(wb_cam)
        for ii in range(b):
            cam_fea = wb_cam[ii].unsqueeze(0)
            cam_wei_inv = tran_weight[ii].inverse()
            wb_out[ii] = F.conv2d(cam_fea,cam_wei_inv,groups=48)
        # wb_out = self.actnorm.reverse(wb_out)
        "mapping"
        for i, block in enumerate(self.blocks[::-1]):
            wb_out = block.reverse(wb_out)
        return wb_out

class Camera_Glow_norev(nn.Module):
    def __init__(self, in_channel, cam_num, n_flow,n_block, affine=True, conv_lu=True):
        super().__init__()
        "n_flow=8, n_block=2"


        "image"
        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu))
            n_channel *= 4

        self.blocks.append(Block(n_channel, n_flow, affine=affine))

        "48?"
        self.cam = camera_conv2d_rever(cam_num,48)
        # self.cam = camera_conv2d(cam_num,48)
        self.wb = AdaIN_wb(num_features=48)
        # self.actnorm = ActNorm(48)


    def forward(self, input,camidx, forward=True):
        ""
        if forward:
            return self._forward(input,camidx)
        else:
            return self._reverse(input,camidx)

    def _forward(self, input,camidx):
        "camera trans weights"
        tran_weight = self.cam(camidx)
        tran_weight, _ = torch.linalg.qr(tran_weight)
        tran_weight = torch.reshape(tran_weight,(-1,48,1,1,1))
        b = tran_weight.shape[0]
        # tran_weight = tran_weight.unsqueeze(2).unsqueeze(3)
        "common mapping"
        z = input
        for block in self.blocks:
            z = block(z)

        # z = self.actnorm(z)
        z_cam = torch.zeros_like(z)
        for ii in range(b):
            cam_fea = z[ii].unsqueeze(0)
            cam_wei = tran_weight[ii]
            z_cam[ii] = F.conv2d(cam_fea,cam_wei,groups=48)

        return z_cam,tran_weight

    def _reverse(self, z_cam,tran_weight):
        out = z_cam
        out = self.wb(out)
        for i, block in enumerate(self.blocks[::-1]):
            out = block.reverse(out)
        return out

class Camera_Glow_norev_re(nn.Module):
    def __init__(self, in_channel, cam_num, n_flow,n_block, affine=True, conv_lu=True):
        super().__init__()
        "n_flow=8, n_block=2"

        self.gamma = 2.2
        "image"
        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow, affine=affine,
                                     conv_lu=conv_lu))
            n_channel *= 4

        self.blocks.append(Block(n_channel, n_flow, affine=affine))

        "48?"
        # self.cam = camera_conv2d_rever(cam_num,48)

        self.wb = AdaIN_wb(num_features=48)
        # self.actnorm = ActNorm(48)


    def forward(self, input,camidx, forward=True):
        ""
        if forward:
            return self._forward(input,camidx)
        else:
            return self._reverse(input,camidx)

    def _forward(self, input,camidx):
        "camera trans weights"
        "sRGB线性化"
        # input = input.pow(self.gamma)

        # tran_weight = tran_weight.unsqueeze(2).unsqueeze(3)
        "common mapping"
        z = input
        for block in self.blocks:
            z = block(z)
        return z

    def _reverse(self, z_cam,camidx):
        ""
        "white-balance"
        z_cam = self.wb(z_cam)

        "camera transformation"
        tran_weight = self.cam(camidx)
        tran_weight, _ = torch.linalg.qr(tran_weight)
        tran_weight = torch.reshape(tran_weight, (-1, 48, 1, 1, 1))
        b = tran_weight.shape[0]

        out_cam = torch.zeros_like(z_cam)
        for ii in range(b):
            cam_fea = z_cam[ii].unsqueeze(0)
            cam_wei = tran_weight[ii]
            out_cam[ii] = F.conv2d(cam_fea, cam_wei, groups=48)

        "projection"
        # out = out_cam

        out = z_cam
        for i, block in enumerate(self.blocks[::-1]):
            out = block.reverse(out)

        return out

class Camera_Glow_wbwithcam(nn.Module):
    def __init__(self, in_channel, cam_num, n_flow,n_block, affine=True, conv_lu=True):
        super().__init__()
        "n_flow=8, n_block=2"

        self.gamma = 2.2
        "image"
        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow, affine=affine,
                                     conv_lu=conv_lu))
            n_channel *= 4

        self.blocks.append(Block(n_channel, n_flow, affine=affine))

        "48?"
        self.wb = AdaIN_wb_cam(num_features=48,cam_num=cam_num)
        # self.cam = camera_conv2d_rever(cam_num,48)
        # self.cam = camera_conv2d(cam_num,48)
        # self.wb = AdaIN_wb(num_features=48)
        # self.actnorm = ActNorm(48)



    def forward(self, input,camidx, forward=True):
        ""
        if forward:
            return self._forward(input,camidx)
        else:
            return self._reverse(input,camidx)

    def _forward(self, input,camidx):
        "common mapping"
        z = input
        for block in self.blocks:
            z = block(z)
        return z

    def _reverse(self, z_cam,camidx):
        ""
        "white-balance"
        out_cam = self.wb(z_cam,camidx)

        "projection"
        out = out_cam
        for i, block in enumerate(self.blocks[::-1]):
            out = block.reverse(out)

        return out



class Glow(nn.Module):
    def __init__(self, in_channel, n_flow, n_block, affine=True, conv_lu=True):
        super().__init__()
        "n_flow=8, n_block=2"

        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu))
            n_channel *= 4

        self.blocks.append(Block(n_channel, n_flow, affine=affine))

        "48?"
        self.wb = AdaIN_wb(num_features=48)
        # self.wb = AdaIN_wb(num_features=48)

        # self.wb = AdaIN_wb_feat(num_features=48)


    def forward(self, input, forward=True, style=None):
        ""
        if forward:
            return self._forward(input)
        else:
            return self._reverse(input)

    def _forward(self, input):
        z = input
        for block in self.blocks:
            z = block(z)
        return z

    def _reverse(self, z):
        out = z
        out = self.wb(out)
        for i, block in enumerate(self.blocks[::-1]):
            out = block.reverse(out)
        return out

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

class ColorCorrection(nn.Module):
    def __init__(self, num_features,camera_dim):
        super().__init__()
        "输入: num_features * 1 *1, 输出: num_features * 1 *1"

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
