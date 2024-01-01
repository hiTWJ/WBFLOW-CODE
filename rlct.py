import matplotlib.pyplot as plt
import numpy as np
import argparse

from unet import deepWBnet
from glow_wb import Camera_Glow_norev_re
import cv2
import torch
import os
from PIL import Image
import matplotlib.image as mpimg
from sklearn.linear_model import LinearRegression
from evaluation.evaluate_cc import evaluate_cc
import seaborn as sns


def vis_feature_(z_c, vis_dir, name):
    NN = []
    for i in range(48):
        save_name = vis_dir + name + '_' + str(i) + '_map.png'
        npimg = z_c[0].cpu().detach().numpy()[i]
        npimg_ = (npimg - np.min(npimg)) / (np.max(npimg) - np.min(npimg))
        NN.append(np.sum(npimg_))
        # plt输入需要时ndarray
        # fig = plt.figure()
        # ",vmin=-15,vmax=15"
        # ", vmin=0, vmax=1"
        # sns_plot = sns.heatmap(npimg,vmin=-10,vmax=10, cmap='YlGnBu')
        # fig.savefig(save_name, bbox_inches='tight')  # 减少边缘空白
        # plt.show()
    return NN


def vis_con_colo_(z_c, vis_dir, name):
    AVG,VAR,CO = [],[],[]
    for i in range(48):
        save_name = vis_dir + name + '_' + str(i) + '_content.png'
        npimg = z_c[0].cpu().detach().numpy()[i]
        # npimg_ = (npimg - np.min(npimg)) / (np.max(npimg) - np.min(npimg))
        # plt输入需要时ndarray
        avg = np.mean(npimg,keepdims=True)
        var = np.var(npimg,keepdims=True)

        AVG.append(avg[0,0])
        VAR.append(var[0,0])

        cont = (npimg-avg)/var
        cont_v = np.sum(cont)
        CO.append(cont_v)
        # fig = plt.figure()
        # ",vmin=-15,vmax=15"
        # ",vmin=0,vmax=1"
        # sns_plot = sns.heatmap(cont,vmin=-10,vmax=10,cmap='YlGnBu')
        # fig.savefig(save_name, bbox_inches='tight')  # 减少边缘空白
        # plt.show()

    return AVG,VAR,CO

def vis_img(imgs,vis_dir,name):
    # imgs = imgs[0]
    # imgs = make_grid(imgs,1)
    save_name = vis_dir+name+'.png'
    # print(save_name)

    npimg = imgs.cpu().detach().numpy()[0]
    npimg = np.transpose(npimg, (1, 2, 0)) # plt输入需要时ndarray
    # fig = plt.figure()
    # plt.imshow(npimg)
    # plt.show()
    # fig = plt.figure()
    # sns_plot = sns.heatmap(npimg, cmap=cm.Blues)
    mpimg.imsave(save_name, (npimg * 255).astype('uint8'))
    # fig.savefig(save_name, bbox_inches='tight') # 减少边缘空白
    # plt.show()


parser = argparse.ArgumentParser()
# Basic options

img_dir_list = ['/dataset/colorconstancy/NUS_CAM2/101/PanasonicGX1_0093_75_AS.jpg',
                '/dataset/colorconstancy/NUS_CAM2/101/Canon1DsMkIII_0175_75_AS.jpg',
                '/dataset/colorconstancy/NUS_CAM2/101/OlympusEPL6_0097_75_AS.jpg']

# gt_dir_list = [ '/dataset/colorconstancy/NUS_CAM2/101/PanasonicGX1_0093_G_AS.jpg',
#                  '/dataset/colorconstancy/NUS_CAM2/101/Canon1DsMkIII_0175_G_AS.jpg',
#                 '/dataset/colorconstancy/NUS_CAM2/101/OlympusEPL6_0097_G_AS.jpg',]


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
"GLOW_WB_noconv_feat, GLOW_WB_CAMTRANS_GCONS"
parser.add_argument('--train_cam',
                    default='UNET_R', type=str, help='')

args = parser.parse_args()

"1.确定可视化的图像路径 2.测试图像 3.输入/输出图像的RGB直方图 4."
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

"model"

net_glow = Camera_Glow_norev_re(3,12,args.n_flow,args.n_block, args.affine, not args.no_lu).to(device)
print("--------loading checkpoint----------")
checkpoint = torch.load('/home/lcx/artflow/output/model/'+'GLOW_WB_RE_12000'+'/'+'best_model.tar')
net_glow.load_state_dict(checkpoint['state_dict'])
net_glow.eval()

camera_list = ["Canon1DsMkIII", "Canon600D", "FujifilmXM1","NikonD5200", "OlympusEPL6", "PanasonicGX1",
               "SamsungNX2000", "SonyA57"]

ERROR = np.zeros((8,35))


save_dir = '/dataset/colorconstancy/output/'
pair_dir = '/dataset/colorconstancy/NUS_CAM1/3/'
vis_dir ='/home/lcx/artflow/output/vis1/'
pos = '.jpg'

AA,VV,CC,FF = [],[],[],[]
for fn in os.listdir(pair_dir):
    if fn.lower().endswith(pos) and fn.split("_")[0]== 'Canon1DsMkIII' and not (fn.split("_")[-2] == 'G'):
        "cam_label"
        print(fn)
        sp = fn.split("_")
        ct = sp[-2]
        cam = sp[0]
        "gt"
        gt_fn = fn.replace(sp[-2],'G')
        gt = Image.open(pair_dir + gt_fn)
        gt_resized = np.array(gt.resize((256, 256)))
        gt = (np.array(gt) / 255).astype(np.float32)
        # plt.imshow(image)
        # plt.show()
        # image_resized = np.array(image_resized)
        # img = image_resized.transpose((2, 0, 1))
        gt_resized = np.array(gt_resized) / 255
        gt = gt_resized.transpose((2, 0, 1))
        gt = torch.from_numpy(gt)
        gt = gt.unsqueeze(0)
        gt = gt.to(device=device, dtype=torch.float32)

        cam_label = np.zeros((12, 1, 1))
        cam_label[camera_list.index(cam)]=1
        "img"
        image = Image.open(pair_dir+fn)
        image_resized = np.array(image.resize((256, 256)))
        image = (np.array(image) / 255).astype(np.float32)
        # plt.imshow(image)
        # plt.show()
        # image_resized = np.array(image_resized)
        # img = image_resized.transpose((2, 0, 1))
        image_resized = np.array(image_resized) / 255
        img = image_resized.transpose((2, 0, 1))
        img = img
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        img = img.to(device=device, dtype=torch.float32)

        cam_label = torch.from_numpy(cam_label)
        cam_label = cam_label.unsqueeze(0)
        cam_label = cam_label.to(device=device, dtype=torch.float32)

        "network"
        z_c = net_glow(img, cam_label, forward=True)
        output = net_glow(z_c, cam_label, forward=False)


        # vis_img(img, vis_dir, fn.split('.')[0]+'_input')
        # vis_img(output, vis_dir, fn.split('.')[0] + '_output')
        # vis_img(gt, vis_dir, fn.split('.')[0] + '_gt')

        F = vis_feature_(z_c, vis_dir, fn.split('.')[0])
        FF.append(F)

        "content"
        AVG,VAR,CO=vis_con_colo_(z_c, vis_dir, fn.split('.')[0])
        AA.append(AVG)
        VV.append(VAR)
        CC.append(CO)

AA = np.array(AA)
VV = np.array(VV)
CC = np.array(CC)
FF = np.array(FF)
print('d')







