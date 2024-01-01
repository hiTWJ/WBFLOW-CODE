import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
from pathlib import Path
import matplotlib.pyplot as plt
from evaluation.evaluate_cc import evaluate_cc
from torchvision.utils import save_image,make_grid
import seaborn as sns
sns.set()

from deep_wb_single_task import deepWBnet
from matplotlib import cm

import time
import numpy as np
import random
from glow_wb import Glow, Camera_Glow

from dataset_msa import BasicDataset
from torch.utils.data import DataLoader, random_split




class Evaluator:
    def __init__(self, args, device):
        self.args = args
        self.cam_list = ["Canon1DsMkIII", "Canon600D", "FujifilmXM1", "NikonD40", "NikonD5200", "OlympusEPL6",
                         "PanasonicGX1",
                         "SamsungNX2000", "SonyA57"]
        # self.config = config
        self.device = device
        self.global_step = 0
        self.bs = 1

        # networks
        # self.type = 'GLOW_WB_noconv_feat'
        # self.glow = Glow(3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu).to(device)
        # self.glow = Camera_Glow(3, 8, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu).to(device)

        # print("--------loading checkpoint----------")
        # checkpoint = torch.load('/home/lcx/artflow/output/model/'+self.type+'/'+'best_model.tar')
        # args.start_iter = checkpoint['iter']
        # self.glow.load_state_dict(checkpoint['state_dict'])

        self.type = args.train_cam
        print(self.type)
        self.unet = deepWBnet().to(device)
        print("--------loading checkpoint----------")
        checkpoint = torch.load('/home/lcx/artflow/output/model/UNET_12000/best_model.tar')
        # self.unet.load_state_dict(checkpoint)
        # checkpoint = torch.load('/home/lcx/deep_final/deep_final/PyTorch/models/' + self.type + '/' + 'best_model.tar')
        self.unet.load_state_dict(checkpoint['state_dict'])

        # dataloaders
        "Cube: 相同场景，不同色温测试图像"
        self.pz = 256

        val_can1_dataset = BasicDataset(name='Canon1DsMkIII_ct_5', patch_size=256, patch_num_per_image=1, type='val')
        self.val_can1 = DataLoader(val_can1_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        val_can6_dataset = BasicDataset(name='Canon600D_ct_5', patch_size=256, patch_num_per_image=1, type='val')
        self.val_can6 = DataLoader(val_can6_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        val_fuj_dataset = BasicDataset(name='FujifilmXM1_ct_5', patch_size=256, patch_num_per_image=1, type='val')
        self.val_fuj = DataLoader(val_fuj_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        val_nik_dataset = BasicDataset(name='NikonD5200_ct_5', patch_size=256, patch_num_per_image=1, type='val')
        self.val_nik = DataLoader(val_nik_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        val_sam_dataset = BasicDataset(name='SamsungNX2000_ct_5', patch_size=256, patch_num_per_image=1, type='val')
        self.val_sam = DataLoader(val_sam_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        val_pan_dataset = BasicDataset(name='PanasonicGX1_ct_5', patch_size=256, patch_num_per_image=1, type='val')
        self.val_pan = DataLoader(val_pan_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        val_sony_dataset = BasicDataset(name='SonyA57_ct_5', patch_size=256, patch_num_per_image=1, type='val')
        self.val_sony = DataLoader(val_sony_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        val_oly_dataset = BasicDataset(name='OlympusEPL6_ct_5', patch_size=256, patch_num_per_image=1, type='val')
        self.val_oly = DataLoader(val_oly_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        "txt"
        self.error_txt = '/home/lcx/artflow/error/' + self.type+'_Deltae.txt'

        try:
            f = open(self.error_txt, 'r')
            f.close()
        except IOError:
            f = open(self.error_txt, 'w')


    def error_evaluation(self,error_list):
        es = np.array(error_list)
        es.sort()
        ae = np.array(es).astype(np.float32)

        x, y, z = np.percentile(ae, [25, 50, 75])
        Mean = np.mean(ae)
        # Med = np.median(ae)
        # Tri = (x+ 2 * y + z)/4
        # T25 = np.mean(ae[:int(0.25 * len(ae))])
        # L25 = np.mean(ae[int(0.75 * len(ae)):])

        print("Mean\tQ1\tQ2\tQ3")
        print("{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(Mean, x, y, z))

    def save_e(self,val_loader_s_c,type):
        "type:相机type"
        D_sc, MS_sc, MA_sc = self.do_eval(val_loader_s_c, self.device)
        "numpy array 格式：1.存 2.求指标"
        print(type)
        self.error_evaluation(D_sc)
        self.error_evaluation(MS_sc)
        self.error_evaluation(MA_sc)

        SC = {}
        SC['DE'] = D_sc
        SC['MS'] = MS_sc
        SC['MA'] = MA_sc
        np.save('/home/lcx/artflow/error/' + self.type + '_'+ type, SC)



    def do_eval(self, loader,device,CAM=False):
        visdir = './output/vis/'
        if not(os.path.exists(visdir)):
            os.makedirs(visdir)

        D,MS,MA = [],[],[]

        for batch in loader:
            "image"
            gt = batch['gt-AWB'].to(device=device, dtype=torch.float32)
            imgs = batch['image'].to(device=device, dtype=torch.float32)
            "label"
            name = batch['name']
            # print(name)
            # ct = batch['ct'][0].to(device=device, dtype=torch.float32)
            cam = batch['label'].to(device=device, dtype=torch.float32)

            "只保留名称"
            name_ = name[0][0]
            cam_name = name_.split('_')[0]
            visdir_ = './output/vis/'+cam_name+'/'
            if not(os.path.exists(visdir_)):
                os.makedirs(visdir_)

            # content/style ---> z ---> stylized
            # z_c,z_cam = self.glow(imgs,cam, forward=True)
            # out_img,_ = self.glow(z_c,z_cam, forward=False)

            out_img = self.unet(imgs)

            # if CAM:
            #     z_c,z_cam = self.glow(imgs,cam, forward=True)
            #     out_img,_ = self.glow(z_c,z_cam, forward=False)
            # else :
            #     z_c = self.glow(imgs, forward=True)
            #     out_img = self.glow(z_c, forward=False)


            ""
            gt_ = gt.cpu().detach().numpy()[0].transpose(1,2,0)
            out_img_ =out_img.cpu().detach().numpy()[0].transpose(1,2,0)

            "vis"
            # self.vis_img(imgs,visdir_,name_)
            # self.vis_feature(z_c,visdir_,name_)

            deltaE00, MSE, MAE = evaluate_cc(out_img_ * 255, gt_ * 255, 0, opt=3)

            "存txt"
            error_save = open(self.error_txt, mode='a')
            error_save.write(
                '\n' + 'Name:' + str(name_)  + '  DELTA E:' + str(round(deltaE00,3))+ '  MSE:' + str(round(MSE[0],3))+ '  MAE:' + str(round(MAE,3)))
            # 关闭文件
            error_save.close()
            D.append(deltaE00)
            MS.append(MSE[0])
            MA.append(MAE)

        D = np.array(D)
        MS = np.array(MS)
        MA = np.array(MA)

        return D,MS,MA

    def vis_feature(self,z_c,vis_dir,name):
        z_c_ = torch.transpose(z_c, 0, 1).repeat(1, 3, 1, 1)
        z_vis = make_grid(z_c_, nrow=8, padding=2)

        save_name = vis_dir+name+'_z_c.png'
        # print(save_name)

        npimg = z_vis.cpu().numpy()
        npimg = np.transpose(npimg, (1, 2, 0))[:, :, 0]  # plt输入需要时ndarray
        fig = plt.figure()
        sns_plot = sns.heatmap(npimg,vmin=-15,vmax=15)
        fig.savefig(save_name, bbox_inches='tight') # 减少边缘空白
        # plt.show()


    def vis_img(self, imgs,vis_dir,name):
        imgs = imgs[0]
        # imgs = make_grid(imgs,1)
        save_name = vis_dir+name+'_img.png'
        # print(save_name)

        npimg = imgs.cpu().numpy()
        npimg = np.transpose(npimg, (1, 2, 0)) # plt输入需要时ndarray
        fig = plt.figure()
        plt.imshow(npimg)
        # fig = plt.figure()
        # sns_plot = sns.heatmap(npimg, cmap=cm.Blues)
        fig.savefig(save_name, bbox_inches='tight') # 减少边缘空白
        # plt.show()


    def do_testing(self):
        # self.glow.eval()
        self.unet.eval()

        with torch.no_grad():
            print('--------------------------------')
            self.save_e(self.val_can1, type='Canon1DsMkIII')
            print('--------------------------------')
            self.save_e(self.val_can6, type='Canon600D')
            print('--------------------------------')
            self.save_e(self.val_fuj, type='FujifilmXM1')
            print('--------------------------------')
            self.save_e(self.val_nik, type='NikonD5200')
            print('--------------------------------')
            self.save_e(self.val_oly, type='OlympusEPL6')
            print('--------------------------------')
            self.save_e(self.val_pan, type='PanasonicGX1')
            print('--------------------------------')
            self.save_e(self.val_sam, type='SamsungNX2000')
            print('--------------------------------')
            self.save_e(self.val_sony, type='SonyA57')

    def test_transform(self,img, size):
        transform_list = []
        h, w, _ = np.shape(img)
        if h < w:
            newh = size
            neww = w / h * size
        else:
            neww = size
            newh = h / w * size
        neww = int(neww // 4 * 4)
        newh = int(newh // 4 * 4)
        transform_list.append(transforms.Resize((newh, neww)))
        transform_list.append(transforms.ToTensor())
        transform = transforms.Compose(transform_list)
        return transform

def main():
    parser = argparse.ArgumentParser()
    # Basic options

    # Additional options
    parser.add_argument('--size', type=int, default=256,
                        help='New size for the content and style images, \
                        keeping the original size if set to 0')
    parser.add_argument('--crop', action='store_true',
                        help='do center crop to create squared image')
    parser.add_argument('--save_ext', default='.jpg',
                        help='The extension name of the output image')
    parser.add_argument('--output', type=str, default='output',
                        help='Directory to save the output image(s)')

    # glow parameters
    parser.add_argument('--operator', type=str, default='wct',
                        help='style feature transfer operator')
    parser.add_argument('--n_flow', default=8, type=int, help='number of flows in each block')  # 32
    parser.add_argument('--n_block', default=2, type=int, help='number of blocks')  # 4
    parser.add_argument('--no_lu', action='store_true', help='use plain convolution instead of LU decomposed version')
    parser.add_argument('--affine', default=False, type=bool, help='use affine coupling instead of additive')
    parser.add_argument('--train_cam',
                        default='deepwb',
                        type=str, help='')

    args = parser.parse_args()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    print('------------Camera Testing---------')

    evaluator = Evaluator(args, device)
    evaluator.do_testing()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
    print('DONE!!!')
