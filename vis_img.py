import matplotlib.pyplot as plt
import numpy as np
import argparse
import WBsRGB as wb_srgb
from unet import deepWBnet
from glow_wb import Camera_Glow_norev_re
import cv2
import torch
import os
from PIL import Image
import matplotlib.image as mpimg
from sklearn.linear_model import LinearRegression
from evaluation.evaluate_cc import evaluate_cc

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

parser = argparse.ArgumentParser()
# Basic options

# img_dir_list = ['/dataset/colorconstancy/NUS_CAM2/101/PanasonicGX1_0093_75_AS.jpg',
#                 '/dataset/colorconstancy/NUS_CAM2/101/Canon1DsMkIII_0175_75_AS.jpg',
#                 '/dataset/colorconstancy/NUS_CAM2/101/OlympusEPL6_0097_75_AS.jpg']
#
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
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

"model"
upgraded_model = 0
gamut_mapping = 2
net_wb = wb_srgb.WBsRGB(gamut_mapping=gamut_mapping,
                         upgraded=upgraded_model)
#
# net_nut = deepWBnet().to(device)
# print("--------loading checkpoint----------")
# checkpoint = torch.load('/home/lcx/artflow/output/model/'+'UNET_R'+'/'+'best_model.tar')
# net_nut.load_state_dict(checkpoint['state_dict'])
# net_nut.eval()

# net_glow = Camera_Glow_norev_re(3,12,args.n_flow,args.n_block, args.affine, not args.no_lu).to(device)
# print("--------loading checkpoint----------")
# checkpoint = torch.load('/home/lcx/artflow/output/model/'+'GLOW_WB_RE'+'/'+'best_model.tar')
# net_glow.load_state_dict(checkpoint['state_dict'])
# net_glow.eval()

camera_list = ["Canon1DsMkIII", "Canon600D", "FujifilmXM1","NikonD5200", "OlympusEPL6", "PanasonicGX1",
               "SamsungNX2000", "SonyA57"]

ERROR = np.zeros((8,35))

for jj in range(len(camera_list)):
    cam = camera_list[jj]
    data = np.load(f'/home/lcx/deep_final/deep_final/folds/val_' + cam + '_sample.npy',
                  allow_pickle=True).item()
    img_dir_list = data['name']
    gt_dir_list = data['gt']


    save_dir = '/dataset/colorconstancy/output/'

    for ii in range(len(img_dir_list)):
        in_dir = img_dir_list[ii]
        gt_dir = gt_dir_list[ii]
        name = os.path.split(in_dir)[1]
        name_ = name.split(".")[0]
        print(name_)
        cam_label = np.zeros((12, 1, 1))
        "Oly:5 Canon1DsMkIII:0"
        cam_label[5, 0, 0] = 1
        cam_label = torch.from_numpy(cam_label).to(device=device, dtype=torch.float32)
        cam_label = cam_label.unsqueeze(0)

        image = Image.open(in_dir)
        image_resized = np.array(image.resize((256,256)))
        image = (np.array(image)/255).astype(np.float32)
        # image_resized = np.array(image_resized)
        # img = image_resized.transpose((2, 0, 1))
        image_resized = np.array(image_resized)/255
        img = image_resized.transpose((2, 0, 1))
        img = img
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        img = img.to(device=device, dtype=torch.float32)

        # mpimg.imsave(save_dir + 'INPUT'+str(ii+2)+'.jpg', (image * 255).astype('uint8'))
        # "his"
        # count_input = cale_his(image)/(image.shape[0]*image.shape[1])
        #
        gt = np.array(Image.open(gt_dir))
        gt = (gt / 255).astype(np.float32)
        # mpimg.imsave(save_dir + 'GT'+str(ii+2)+'.jpg', (gt * 255).astype('uint8'))
        # count_gt = cale_his(gt) / (gt.shape[0] * gt.shape[1])

        # "knn-wb"
        output1 = net_wb.correctImage(image)
        deltaE00 = evaluate_cc(output1 * 255, gt * 255, 0, opt=1)
        ERROR[jj, ii] = deltaE00
        mpimg.imsave(save_dir +name_+ '_KNN'+'.jpg', (output1 * 255).astype('uint8'))
        # count_output1 = cale_his(output1.astype(np.float32)) / (output1.shape[0] * output1.shape[1])
        #
        # "dwb"
        # output2 = net_nut(img)
        # output2_ = output2[0].cpu().detach().numpy().transpose((1,2, 0))
        # m_awb = get_mapping_func(image_resized, output2_)
        # output_awb = outOfGamutClipping(apply_mapping_func(image, m_awb))
        # deltaE00 = evaluate_cc(output_awb * 255, gt * 255, 0, opt=1)
        # ERROR[jj, ii] = deltaE00
        # # plt.imshow(output_awb)
        # # plt.show()
        # mpimg.imsave(save_dir +name_+ '_DWB'+'.jpg', (output_awb * 255).astype('uint8'))
        # count_output2 = cale_his(output_awb.astype(np.float32)) / (output_awb.shape[0] * output_awb.shape[1])

        "glow"
        # z_c = net_glow(img,cam_label, forward=True)
        # output3 = net_glow(z_c,cam_label, forward=False)
        # output3_ = output3[0].cpu().detach().numpy().transpose((1, 2, 0))
        # m_awb1 = get_mapping_func(image_resized, output3_)
        # output_awb1 = outOfGamutClipping(apply_mapping_func(image, m_awb1))

        # deltaE00 = evaluate_cc(output_awb * 255, gt * 255, 0, opt=1)
        # ERROR[jj,ii] = deltaE00

        # plt.imshow(output_awb1)
        # plt.show()
        # mpimg.imsave(save_dir +name_+ '_DWB'+'.jpg', (output_awb1 * 255).astype('uint8'))
        # count_output3 = cale_his(output_awb1.astype(np.float32)) / (output_awb1.shape[0] * output_awb1.shape[1])

print('d')
