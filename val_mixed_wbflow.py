import random

import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import os
from glow_wb import Glow,Camera_Glow_norev_re,Camera_Glow_wbwithcam
from sklearn.manifold import TSNE
from err import err_compution
import argparse

EPS = 1e-6




class RGBuvHistBlock(nn.Module):
    def __init__(self, h=64, insz=150, resizing='interpolation',
                 method='inverse-quadratic', sigma=0.02, intensity_scale=True,
                 device='cuda:0'):

        """ Computes the RGB-uv histogram feature of a given image.
        Args:
          h: histogram dimension size (scalar). The default value is 64.
          insz: maximum size of the input image; if it is larger than this size, the
            image will be resized (scalar). Default value is 150 (i.e., 150 x 150
            pixels).
          resizing: resizing method if applicable. Options are: 'interpolation' or
            'sampling'. Default is 'interpolation'.
          method: the method used to count the number of pixels for each bin in the
            histogram feature. Options are: 'thresholding', 'RBF' (radial basis
            function), or 'inverse-quadratic'. Default value is 'inverse-quadratic'.
          sigma: if the method value is 'RBF' or 'inverse-quadratic', then this is
            the sigma parameter of the kernel function. The default value is 0.02.
          intensity_scale: boolean variable to use the intensity scale (I_y in
            Equation 2). Default value is True.

        Methods:
          forward: accepts input image and returns its histogram feature. Note that
            unless the method is 'thresholding', this is a differentiable function
            and can be easily integrated with the loss function. As mentioned in the
             paper, the 'inverse-quadratic' was found more stable than 'RBF' in our
             training.
        """
        super(RGBuvHistBlock, self).__init__()

        self.h = h
        self.insz = insz
        self.device = device
        self.resizing = resizing
        self.method = method
        self.intensity_scale = intensity_scale
        if self.method == 'thresholding':
            self.eps = 6.0 / h
        else:
            self.sigma = sigma

    def forward(self, x):
        device = torch.device('cuda:0')
        x = torch.clamp(x, 0, 1)
        if x.shape[2] > self.insz or x.shape[3] > self.insz:
            if self.resizing == 'interpolation':
                x_sampled = F.interpolate(x, size=(self.insz, self.insz),
                                          mode='bilinear', align_corners=False)
            elif self.resizing == 'sampling':
                inds_1 = torch.LongTensor(
                    np.linspace(0, x.shape[2], self.h, endpoint=False)).to(
                    device=self.device)
                inds_2 = torch.LongTensor(
                    np.linspace(0, x.shape[3], self.h, endpoint=False)).to(
                    device=self.device)
                x_sampled = x.index_select(2, inds_1)
                x_sampled = x_sampled.index_select(3, inds_2)
            else:
                raise Exception(
                    f'Wrong resizing method. It should be: interpolation or sampling. '
                    f'But the given value is {self.resizing}.')
        else:
            x_sampled = x

        L = x_sampled.shape[0]  # size of mini-batch
        if x_sampled.shape[1] > 3:
            x_sampled = x_sampled[:, :3, :, :]
        X = torch.unbind(x_sampled, dim=0)
        hists = torch.zeros((x_sampled.shape[0], 3, self.h, self.h))
        # hists = hists.to(device=self.device)
        hists = hists.to(device)
        for l in range(L):
            I = torch.t(torch.reshape(X[l], (3, -1)))
            II = torch.pow(I, 2)
            if self.intensity_scale:
                Iy = torch.unsqueeze(torch.sqrt(II[:, 0] + II[:, 1] + II[:, 2] + EPS),
                                     dim=1)
            else:
                Iy = 1

            Iy = Iy.to(self.device)

            Iu0 = torch.unsqueeze(torch.log(I[:, 0] + EPS) - torch.log(I[:, 1] + EPS),
                                  dim=1).to(self.device)
            Iv0 = torch.unsqueeze(torch.log(I[:, 0] + EPS) - torch.log(I[:, 2] + EPS),
                                  dim=1).to(self.device)
            diff_u0 = abs(
                Iu0 - torch.unsqueeze(torch.tensor(np.linspace(-3, 3, num=self.h)),
                                      dim=0).to(self.device))
            diff_v0 = abs(
                Iv0 - torch.unsqueeze(torch.tensor(np.linspace(-3, 3, num=self.h)),
                                      dim=0).to(self.device))
            if self.method == 'thresholding':
                diff_u0 = torch.reshape(diff_u0, (-1, self.h)) <= self.eps / 2
                diff_v0 = torch.reshape(diff_v0, (-1, self.h)) <= self.eps / 2
            elif self.method == 'RBF':
                diff_u0 = torch.pow(torch.reshape(diff_u0, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_v0 = torch.pow(torch.reshape(diff_v0, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_u0 = torch.exp(-diff_u0)  # Radial basis function
                diff_v0 = torch.exp(-diff_v0)
            elif self.method == 'inverse-quadratic':
                diff_u0 = torch.pow(torch.reshape(diff_u0, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_v0 = torch.pow(torch.reshape(diff_v0, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_u0 = 1 / (1 + diff_u0)  # Inverse quadratic
                diff_v0 = 1 / (1 + diff_v0)
            else:
                raise Exception(
                    f'Wrong kernel method. It should be either thresholding, RBF,'
                    f' inverse-quadratic. But the given value is {self.method}.')
            diff_u0 = diff_u0.type(torch.float32)
            diff_v0 = diff_v0.type(torch.float32)
            a = torch.t(Iy * diff_u0)
            hists[l, 0, :, :] = torch.mm(a, diff_v0)

            Iu1 = torch.unsqueeze(torch.log(I[:, 1] + EPS) - torch.log(I[:, 0] + EPS),
                                  dim=1).to(self.device)
            Iv1 = torch.unsqueeze(torch.log(I[:, 1] + EPS) - torch.log(I[:, 2] + EPS),
                                  dim=1).to(self.device)
            diff_u1 = abs(
                Iu1 - torch.unsqueeze(torch.tensor(np.linspace(-3, 3, num=self.h)),
                                      dim=0).to(self.device))
            diff_v1 = abs(
                Iv1 - torch.unsqueeze(torch.tensor(np.linspace(-3, 3, num=self.h)),
                                      dim=0).to(self.device))

            if self.method == 'thresholding':
                diff_u1 = torch.reshape(diff_u1, (-1, self.h)) <= self.eps / 2
                diff_v1 = torch.reshape(diff_v1, (-1, self.h)) <= self.eps / 2
            elif self.method == 'RBF':
                diff_u1 = torch.pow(torch.reshape(diff_u1, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_v1 = torch.pow(torch.reshape(diff_v1, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_u1 = torch.exp(-diff_u1)  # Gaussian
                diff_v1 = torch.exp(-diff_v1)
            elif self.method == 'inverse-quadratic':
                diff_u1 = torch.pow(torch.reshape(diff_u1, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_v1 = torch.pow(torch.reshape(diff_v1, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_u1 = 1 / (1 + diff_u1)  # Inverse quadratic
                diff_v1 = 1 / (1 + diff_v1)

            diff_u1 = diff_u1.type(torch.float32)
            diff_v1 = diff_v1.type(torch.float32)
            a = torch.t(Iy * diff_u1)
            hists[l, 1, :, :] = torch.mm(a, diff_v1)

            Iu2 = torch.unsqueeze(torch.log(I[:, 2] + EPS) - torch.log(I[:, 0] + EPS),
                                  dim=1).to(self.device)
            Iv2 = torch.unsqueeze(torch.log(I[:, 2] + EPS) - torch.log(I[:, 1] + EPS),
                                  dim=1).to(self.device)
            diff_u2 = abs(
                Iu2 - torch.unsqueeze(torch.tensor(np.linspace(-3, 3, num=self.h)),
                                      dim=0).to(self.device))
            diff_v2 = abs(
                Iv2 - torch.unsqueeze(torch.tensor(np.linspace(-3, 3, num=self.h)),
                                      dim=0).to(self.device))
            if self.method == 'thresholding':
                diff_u2 = torch.reshape(diff_u2, (-1, self.h)) <= self.eps / 2
                diff_v2 = torch.reshape(diff_v2, (-1, self.h)) <= self.eps / 2
            elif self.method == 'RBF':
                diff_u2 = torch.pow(torch.reshape(diff_u2, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_v2 = torch.pow(torch.reshape(diff_v2, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_u2 = torch.exp(-diff_u2)  # Gaussian
                diff_v2 = torch.exp(-diff_v2)
            elif self.method == 'inverse-quadratic':
                diff_u2 = torch.pow(torch.reshape(diff_u2, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_v2 = torch.pow(torch.reshape(diff_v2, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_u2 = 1 / (1 + diff_u2)  # Inverse quadratic
                diff_v2 = 1 / (1 + diff_v2)
            diff_u2 = diff_u2.type(torch.float32)
            diff_v2 = diff_v2.type(torch.float32)
            a = torch.t(Iy * diff_u2)
            hists[l, 2, :, :] = torch.mm(a, diff_v2)

        # normalization
        hists_normalized = hists / (
                ((hists.sum(dim=1)).sum(dim=1)).sum(dim=1).view(-1, 1, 1, 1) + EPS)

        return hists_normalized


class expand_greyscale(object):
    def __init__(self, num_channels):
        self.num_channels = num_channels

    def __call__(self, tensor):
        return tensor.expand(self.num_channels, -1, -1)


def histogram_loss(input_hist, target_hist):
    histogram_loss = (1 / np.sqrt(2.0) * (torch.sqrt(torch.sum(
        torch.pow(torch.sqrt(target_hist) - torch.sqrt(input_hist), 2)))) /
                      input_hist.shape[0])

    return histogram_loss

def image_loader(image_name):
    image = Image.open(image_name)
    image = image.resize((256, 256))
    image_np = np.array(image)/255
    # fake batch dimension required to fit network's input dimensions
    image_t = loader(image).unsqueeze(0)
    return image_t.to(device, torch.float), image_np

def tensor_loader(image):
    # image = Image.open(image_name)
    # image_np = np.array(image)/255
    # fake batch dimension required to fit network's input dimensions
    image_t = loader(image).unsqueeze(0)
    return image_t.to(device, torch.float)


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def encode(hist):

    histR_reshaped = np.reshape(np.transpose(hist[:, :, 0]),
                                (1, int(hist.size / 3)), order="F")
    histG_reshaped = np.reshape(np.transpose(hist[:, :, 1]),
                                (1, int(hist.size / 3)), order="F")
    histB_reshaped = np.reshape(np.transpose(hist[:, :, 2]),
                                (1, int(hist.size / 3)), order="F")
    hist_reshaped = np.append(histR_reshaped,
                              [histG_reshaped, histB_reshaped])

    return hist_reshaped

def imsave(tensor, title):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    image.save(title)
    # plt.imshow(image)
    # plt.axis('off')
    # if title is not None:
    #     plt.title(title)
    # plt.pause(0.001)  # pause a bit so that plots are updated

if __name__ == "__main__":
    "ARGS"
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_flow', default=16, type=int, help='number of flows in each block')  # 32
    parser.add_argument('--n_block', default=2, type=int, help='number of blocks')  # 4
    parser.add_argument('--no_lu', action='store_true', help='use plain convolution instead of LU decomposed version')
    parser.add_argument('--affine', default=False, type=bool, help='use affine coupling instead of additive')
    args = parser.parse_args()

    # loss_ = np.load('/home/lcx/sty/output/exp/Mixedscene_within_img.npy')

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    loader = transforms.Compose(
        [transforms.ToTensor()])  # transform it into a torch tensor

    unloader = transforms.ToPILImage()  # reconvert into PIL image

    "模型加载"
    net_glow =Camera_Glow_norev_re(3, 15, args.n_flow, args.n_block, affine=args.affine,
                                     conv_lu=not args.no_lu).to(device)
    print("--------loading checkpoint----------")
    "GLOW_WB_RE_12000_15_32_base: 没有cam"
    "GLOW_WB_RE_12000_15:有cam"
    checkpoint = torch.load('/home/lcx/artflow/output/model/' + 'GLOW_WB_RE_12000_15_32_base' + '/' + '140000.tar')
    net_glow.load_state_dict(checkpoint['state_dict'])
    net_glow.eval()



    "读取cube数据集路径下的图像：相同场景下：两两色温对应，GT就是0"
    "T:2850K, F:3800K, D:5500K, S:7500K"
    # souce_dir = '/dataset/colorconstancy/Cube_input_images/'
    # souce_dir = '/dataset/colorconstancy/Cube_ground_truth_images/'
    souce_dir = '/dataset/colorconstancy/mixedill_test_set_JPG/'
    # ct1 = ['_T_CS.jpg','_F_CS.jpg','_T_CS.jpg','_F_CS.jpg','_T_CS.jpg','_D_CS.jpg','_G_AS.jpg']
    # ct2 = ['_S_CS.jpg','_S_CS.jpg','_D_CS.jpg','_D_CS.jpg','_F_CS.jpg','_S_CS.jpg','_G_AS.jpg']

    ct = ['_T_CS.jpg','_F_CS.jpg','_D_CS.jpg','_S_CS.jpg','_G_AS.jpg']

    save_dir = '/home/lcx/sty/output/exp/mix_withinimg_WBFlow/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    "patch尺寸"
    patch_size = 128

    EM = np.zeros((31, len(ct), 2))
    # target_id = 1

    for i in range(1,31):
        for c in range(len(ct)):
            print(f'CT_P= {str(c)}')
            img_ct_dir = souce_dir +'scene_' +str(i) + ct[c]
            img_gt_dir = souce_dir + 'scene_' + str(i) + '_G_AS.jpg'
            # img_ct2_dir = souce_dir +'scene_' +str(list_ct[i]) + ct2[c]
            if os.path.exists(img_ct_dir):
                print(f'IMG1= {str(i)}')
                img_ct,img_ct_np = image_loader(img_ct_dir)
                img_gt, _ = image_loader(img_gt_dir)

                "白平衡校正"
                "camera_label没有用"
                cam_label = np.zeros((15, 1, 1))
                cam_label[5, 0, 0] = 1
                cam_label = torch.from_numpy(cam_label).to(device=device, dtype=torch.float32)
                cam_label = cam_label.unsqueeze(0)

                "白平衡校正"
                z_c = net_glow(img_ct, cam_label, forward=True)
                img_ct_out = net_glow(z_c, cam_label, forward=False)

                imsave(img_ct_out, save_dir + 'scene_' + str(i) + '_WBFlow_256' + ct[c])

                # imshow(img_ct_out)

                "Delta差异"
                deltae, mse, _ = err_compution(img_gt[0], img_ct_out[0])
                EM[i, c, 0] = deltae
                EM[i, c, 1] = mse



    EM = np.array(EM)
    np.save('/home/lcx/sty/output/exp/Mixedscene_img_WBFlow_256.npy',EM)

    print('d')


















