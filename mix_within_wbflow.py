import random

import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import os
import argparse
from err import err_compution
import cv2
from sklearn.linear_model import LinearRegression

from glow_wb import Glow,Camera_Glow_norev_re,Camera_Glow_wbwithcam
EPS = 1e-6

def get_mapping_func(image1, image2):
    """ Computes the polynomial mapping """
    image1 = np.reshape(image1, [-1, 3])
    image2 = np.reshape(image2, [-1, 3])
    m = LinearRegression().fit(kernelP(image1), image2)
    return m

def kernelP(I):
    """ Kernel function: kernel(r, g, b) -> (r,g,b,rg,rb,gb,r^2,g^2,b^2,rgb,1)
        Ref: Hong, et al., "A study of digital camera colorimetric characterization
         based on polynomial modeling." Color Research & Application, 2001. """
    return (np.transpose((I[:, 0], I[:, 1], I[:, 2], I[:, 0] * I[:, 1], I[:, 0] * I[:, 2],
                          I[:, 1] * I[:, 2], I[:, 0] * I[:, 0], I[:, 1] * I[:, 1],
                          I[:, 2] * I[:, 2], I[:, 0] * I[:, 1] * I[:, 2],
                          np.repeat(1, np.shape(I)[0]))))

def outOfGamutClipping(I):
    """ Clips out-of-gamut pixels. """
    I[I > 1] = 1  # any pixel is higher than 1, clip it to 1
    I[I < 0] = 0  # any pixel is below 0, clip it to 0
    return I

def cale_his(img):
    count_r = cv2.calcHist(img, [0], None, [256], [0.0, 1.0])
    count_g = cv2.calcHist(img, [1], None, [256], [0.0, 1.0])
    count_b = cv2.calcHist(img, [2], None, [256], [0.0, 1.0])
    count = np.concatenate((count_r,count_g,count_b),axis=1)
    return count

def apply_mapping_func(image, m):
    """ Applies the polynomial mapping """
    sz = image.shape
    image = np.reshape(image, [-1, 3])
    result = m.predict(kernelP(image))
    result = np.reshape(result, [sz[0], sz[1], sz[2]])
    return result



class RGBuvHistBlock(nn.Module):
    def __init__(self, h=64, insz=150, resizing='interpolation',
                 method='inverse-quadratic', sigma=0.02, intensity_scale=True,
                 device='cuda:0'):
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


def histogram_loss(input_hist, target_hist):
    histogram_loss = (1 / np.sqrt(2.0) * (torch.sqrt(torch.sum(
        torch.pow(torch.sqrt(target_hist) - torch.sqrt(input_hist), 2)))) /
                      input_hist.shape[0])

    return histogram_loss

def image_loader(image_name):
    image = Image.open(image_name)
    image = image.resize((1024, 512))
    image_np = np.array(image)/255
    # fake batch dimension required to fit network's input dimensions
    image_t = loader(image).unsqueeze(0)
    return image_t.to(device, torch.float), image_np

def image_loader_re(image_name):
    image = Image.open(image_name)
    image1 = image.resize((256, 256))
    image_np1 = np.array(image1) / 255

    image2 = image.resize((1024, 512))
    image_np2 = np.array(image2)/255
    # fake batch dimension required to fit network's input dimensions
    image_t = loader(image1).unsqueeze(0)
    return image_t.to(device, torch.float), image_np1,image_np2



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
    "多光数据，图像内patch之间的色温（uv直方图）差异"
    "以scene为组，30组"
    "以其中一个patch为目标，计算不同色温下，它与其它patches之间的uv直方图的差异"
    "注：由于场景的差异性，不能直接将所有patches加和"

    "ARGS"
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_flow', default=16, type=int, help='number of flows in each block')  # 32
    parser.add_argument('--n_block', default=2, type=int, help='number of blocks')  # 4
    parser.add_argument('--no_lu', action='store_true', help='use plain convolution instead of LU decomposed version')
    parser.add_argument('--affine', default=False, type=bool, help='use affine coupling instead of additive')
    args = parser.parse_args()

    # loss_ = np.load('/home/lcx/sty/output/exp/Mixedscene_within_img.npy')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loader = transforms.Compose(
        [transforms.ToTensor()])  # transform it into a torch tensor

    unloader = transforms.ToPILImage()  # reconvert into PIL image

    "模型加载"
    net_glow =Camera_Glow_norev_re(3, 15, args.n_flow, args.n_block, affine=args.affine,
                                     conv_lu=not args.no_lu).to(device)
    print("--------loading checkpoint----------")
    "GLOW_WB_RE_12000_15_32_base: 没有cam"
    "GLOW_WB_RE_12000_15:有cam"
    checkpoint = torch.load('/home/lcx/artflow/output/model/' + 'GLOW_WB_RE_12000_15_32_base' + '/' + '125000.tar')
    net_glow.load_state_dict(checkpoint['state_dict'])
    net_glow.eval()

    "hist"
    intensity_scale = True
    histogram_size = 32
    max_input_size = 128
    method = 'inverse-quadratic'  # options:'thresholding','RBF','inverse-quadratic'
    # create a histogram block
    histogram_block = RGBuvHistBlock(insz=max_input_size, h=histogram_size,
                                     intensity_scale=intensity_scale,
                                     method=method,
                                     device=device)

    "读取cube数据集路径下的图像：相同场景下：两两色温对应，GT就是0"
    "T:2850K, F:3800K, D:5500K, S:7500K"
    # souce_dir = '/dataset/colorconstancy/Cube_input_images/'
    # souce_dir = '/dataset/colorconstancy/Cube_ground_truth_images/'
    souce_dir = '/dataset/colorconstancy/mixedill_test_set_JPG/'
    # ct1 = ['_T_CS.jpg','_F_CS.jpg','_T_CS.jpg','_F_CS.jpg','_T_CS.jpg','_D_CS.jpg','_G_AS.jpg']
    # ct2 = ['_S_CS.jpg','_S_CS.jpg','_D_CS.jpg','_D_CS.jpg','_F_CS.jpg','_S_CS.jpg','_G_AS.jpg']

    ct = ['_T_CS.jpg','_F_CS.jpg','_D_CS.jpg','_S_CS.jpg','_G_AS.jpg']

    "targetpatch * scene * ct * patches"
    tp_list = [5, 21]
    Hist_loss = np.zeros((len(tp_list), 30, len(ct), 32))
    ERROR = np.zeros((len(tp_list), 30, len(ct)))
    D_loss = np.zeros((len(tp_list), 30, len(ct), 32))

    save_dir = '/home/lcx/sty/output/exp/mix_withinimg_wbflow/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    "patch尺寸"
    patch_size = 128
    EM = []

    "目标块"
    # target_id = 1
    for kk in range(len(tp_list)):
        target_id = tp_list[kk]
        print(f'------------------------TPatch{target_id}------------------------')
        for i in range(1, 31):
            for c in range(len(ct)):
                print(f'CT_P= {str(c)}')
                img_ct_dir = souce_dir + 'scene_' + str(i) + ct[c]
                img_gt_dir = souce_dir + 'scene_' + str(i) + '_G_AS.jpg'
                # img_ct2_dir = souce_dir +'scene_' +str(list_ct[i]) + ct2[c]
                if os.path.exists(img_ct_dir):
                    print(f'IMG1= {str(i)}')
                    img_ct, img_ct_np1,img_ct_np2 = image_loader_re(img_ct_dir)
                    img_gt, _ = image_loader(img_gt_dir)

                    "camera_label没有用"
                    cam_label = np.zeros((15, 1, 1))
                    cam_label[5, 0, 0] = 1
                    cam_label = torch.from_numpy(cam_label).to(device=device, dtype=torch.float32)
                    cam_label = cam_label.unsqueeze(0)

                    "白平衡校正"
                    z_c = net_glow(img_ct, cam_label, forward=True)
                    output3 = net_glow(z_c, cam_label, forward=False)
                    output3_ = output3[0].cpu().detach().numpy().transpose((1, 2, 0))
                    m_awb1 = get_mapping_func(img_ct_np1, output3_)
                    img_ct_out = outOfGamutClipping(apply_mapping_func(img_ct_np2, m_awb1))
                    img_ct_out = tensor_loader(img_ct_out)

                    # imshow(img_ct_out)


                    "存图像"
                    imsave(img_ct_out, save_dir + 'scene_' + str(i) + '_WBFLOW' + ct[c])

                    img_ct_hist = histogram_block(img_ct_out)

                    "分块"
                    img_ct_block = torch.cat(torch.split(img_ct_out, patch_size, 2), 0)
                    img_ct_block = torch.cat(torch.split(img_ct_block, patch_size, 3), 0)

                    img_gt_block = torch.cat(torch.split(img_gt, patch_size, 2), 0)
                    img_gt_block = torch.cat(torch.split(img_gt_block, patch_size, 3), 0)

                    "Delta差异"
                    deltae, _, _ = err_compution(img_gt_block[kk], img_ct_block[kk])
                    ERROR[kk, i - 1, c] = deltae

                    "分块坐标:图像尺寸-1024,512; patch尺寸-128,128"
                    _, _, H, W = tensor_loader(img_ct_np2).shape
                    patch_x = np.array(range(int(patch_size / 2), W, patch_size)).repeat(int(H / patch_size)) / W
                    patch_y = np.array(range(int(patch_size / 2), H, patch_size))[::-1][:, np.newaxis].repeat(
                        int(W / patch_size), 1).flatten('F') / H
                    # patch_y = np.concatenate([patch_y,patch_y,patch_y,patch_y],axis=0)

                    "分块hist"
                    img_ct_block_hist = histogram_block(img_ct_block)

                    # plt.ion()
                    # plt.figure()
                    # imshow(img_ct_block_hist[5] * 100, title='Patch 5')
                    # plt.figure()
                    #
                    # plt.ion()
                    # plt.figure()
                    # imshow(img_ct_block_hist[21] * 100, title='Patch 21')
                    # plt.figure()

                    "1.4.5.6.9.17.20.21.22.25"
                    "存patch"
                    imsave(img_ct_block[1], save_dir + 'scene_' + str(i) + 'P1' + ct[c])
                    imsave(img_ct_block[4], save_dir + 'scene_' + str(i) + 'P4' + ct[c])
                    imsave(img_ct_block[5], save_dir + 'scene_' + str(i) + 'P5' + ct[c])
                    imsave(img_ct_block[6], save_dir + 'scene_' + str(i) + 'P6' + ct[c])
                    imsave(img_ct_block[9], save_dir + 'scene_' + str(i) + 'P9' + ct[c])

                    imsave(img_ct_block[17], save_dir + 'scene_' + str(i) + 'P17' + ct[c])
                    imsave(img_ct_block[20], save_dir + 'scene_' + str(i) + 'P20' + ct[c])
                    imsave(img_ct_block[21], save_dir + 'scene_' + str(i) + 'P21' + ct[c])
                    imsave(img_ct_block[22], save_dir + 'scene_' + str(i) + 'P22' + ct[c])
                    imsave(img_ct_block[25], save_dir + 'scene_' + str(i) + 'P25' + ct[c])

                    "存hist"
                    imsave(img_ct_block_hist[1] * 100, save_dir + 'scene_' + str(i) + 'H1' + ct[c])
                    imsave(img_ct_block_hist[4] * 100, save_dir + 'scene_' + str(i) + 'H4' + ct[c])
                    imsave(img_ct_block_hist[5] * 100, save_dir + 'scene_' + str(i) + 'H5' + ct[c])
                    imsave(img_ct_block_hist[6] * 100, save_dir + 'scene_' + str(i) + 'H6' + ct[c])
                    imsave(img_ct_block_hist[9] * 100, save_dir + 'scene_' + str(i) + 'H9' + ct[c])

                    imsave(img_ct_block_hist[17] * 100, save_dir + 'scene_' + str(i) + 'H17' + ct[c])
                    imsave(img_ct_block_hist[20] * 100, save_dir + 'scene_' + str(i) + 'H20' + ct[c])
                    imsave(img_ct_block_hist[21] * 100, save_dir + 'scene_' + str(i) + 'H21' + ct[c])
                    imsave(img_ct_block_hist[22] * 100, save_dir + 'scene_' + str(i) + 'H22' + ct[c])
                    imsave(img_ct_block_hist[25] * 100, save_dir + 'scene_' + str(i) + 'H25' + ct[c])

                    "Reshape hist"
                    "4,5,20,21 patches"
                    "1,4,5,6,9 patches"
                    "17,20,21,22,25 patches"
                    hist1 = img_ct_block_hist[1].detach().cpu().numpy().transpose((1, 2, 0))
                    hist1 = encode(hist1)
                    EM.append(hist1)

                    hist4 = img_ct_block_hist[4].detach().cpu().numpy().transpose((1, 2, 0))
                    hist4 = encode(hist4)
                    EM.append(hist4)

                    hist5 = img_ct_block_hist[5].detach().cpu().numpy().transpose((1, 2, 0))
                    hist5 = encode(hist5)
                    EM.append(hist5)

                    hist6 = img_ct_block_hist[6].detach().cpu().numpy().transpose((1, 2, 0))
                    hist6 = encode(hist6)
                    EM.append(hist6)

                    hist9 = img_ct_block_hist[9].detach().cpu().numpy().transpose((1, 2, 0))
                    hist9 = encode(hist9)
                    EM.append(hist9)

                    hist17 = img_ct_block_hist[17].detach().cpu().numpy().transpose((1, 2, 0))
                    hist17 = encode(hist17)
                    EM.append(hist17)

                    hist20 = img_ct_block_hist[20].detach().cpu().numpy().transpose((1, 2, 0))
                    hist20 = encode(hist20)
                    EM.append(hist20)

                    hist21 = img_ct_block_hist[21].detach().cpu().numpy().transpose((1, 2, 0))
                    hist21 = encode(hist21)
                    EM.append(hist21)

                    hist22 = img_ct_block_hist[22].detach().cpu().numpy().transpose((1, 2, 0))
                    hist22 = encode(hist22)
                    EM.append(hist22)

                    hist25 = img_ct_block_hist[25].detach().cpu().numpy().transpose((1, 2, 0))
                    hist25 = encode(hist25)
                    EM.append(hist25)

                    # patch_id = np.array(range(0,32))
                    # patch_id_re = np.delete(patch_id,np.where(patch_id==target_id))

                    target_patch_hist = img_ct_block_hist[target_id]
                    target_patch_x = patch_x[target_id]
                    target_patch_y = patch_y[target_id]
                    # plt.ion()
                    # plt.figure()
                    # imshow(img_ct1_hist * 100, title='Input Histogram')
                    # plt.figure()

                    for id in range(32):
                        selected_patch_hist = img_ct_block_hist[id]
                        selected_patch_x = patch_x[id]
                        selected_patch_y = patch_y[id]

                        "距离差异"
                        dis_loss = np.sqrt(np.power((selected_patch_x - target_patch_x), 2) + np.power(
                            (selected_patch_y - target_patch_y), 2))
                        D_loss[kk, i - 1, c, id] = dis_loss
                        print(f'TPatch{target_id}-Patch{id}-Scene{i}-CT{c}-Distance loss = {dis_loss}')
                        "hist差异"
                        histogram_loss = (1 / np.sqrt(2.0) * (torch.sqrt(torch.sum(
                            torch.pow(torch.sqrt(selected_patch_hist) - torch.sqrt(target_patch_hist), 2)))) /
                                          target_patch_hist.shape[0])

                        histogram_loss = histogram_loss.cpu().detach().numpy()
                        Hist_loss[kk, i - 1, c, id] = histogram_loss
                        print(f'TPatch{target_id}-Patch{id}-Scene{i}-CT{c}-Histogram loss = {histogram_loss}')

    EM = np.array(EM)
    Loss = {}
    Loss['hist'] = Hist_loss
    Loss['dis'] = D_loss
    Loss['Target_delta'] = ERROR
    Loss['Re_hist'] = EM
    np.save('/home/lcx/sty/output/exp/Mixedscene_within_img_WBFLOW.npy', Loss)

    print('d')
















