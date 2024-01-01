import matplotlib.pyplot as plt
import numpy as np
import argparse
from glow_wb import Camera_Glow_norev_re
import cv2
import torch
import os
import glob
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

"ARGS"
parser = argparse.ArgumentParser()
parser.add_argument('--n_flow', default=16, type=int, help='number of flows in each block')  # 32
parser.add_argument('--n_block', default=2, type=int, help='number of blocks')  # 4
parser.add_argument('--no_lu', action='store_true', help='use plain convolution instead of LU decomposed version')
parser.add_argument('--affine', default=False, type=bool, help='use affine coupling instead of additive')
args = parser.parse_args()


"Device"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

"Model"
glow = Camera_Glow_norev_re(3, 15, args.n_flow, args.n_block, affine=args.affine,
                                     conv_lu=not args.no_lu)
# glow = Camera_Glow_norev_re(3, 15, 16, 2).to(device)
print("--------loading checkpoint----------")
checkpoint = torch.load('/home/lcx/artflow/output/model/'+'GLOW_WB_RE_12000_15_32_base'+'/'+'140000.tar')
glow.load_state_dict(checkpoint['state_dict'])
glow.to(device)
glow.eval()

"DATA: indoor, single; outdoor, single; indoor, multi"
dir = '/home/lcx/artflow/chuanyin/indoor-multi/'
input_images = glob.glob(os.path.join(dir, '*.jpg'))

"Loop"
for img_name in input_images:
    print(img_name)
    img = Image.open(img_name)
    "输出"
    img_ = np.array(img)

    plt.imshow(img_)
    plt.show()

    "输入"
    img_resized = np.array(img.resize((256, 256)))
    img = (np.array(img_resized) / 255).astype(np.float32)
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    "camera_label没有用"
    cam_label = np.zeros((15, 1, 1))
    cam_label[5, 0, 0] = 1
    cam_label = torch.from_numpy(cam_label).to(device=device, dtype=torch.float32)
    cam_label = cam_label.unsqueeze(0)

    z_c = glow(img,cam_label, forward=True)
    output3 = glow(z_c,cam_label, forward=False)
    output3_ = output3[0].cpu().detach().numpy().transpose((1, 2, 0))
    m_awb1 = get_mapping_func(img_resized, output3_)
    output_awb1 = outOfGamutClipping(apply_mapping_func(img_, m_awb1))

    plt.imshow(output_awb1)
    plt.show()
    save_name = img_name.split('.')[0]+'_wbflow.jpg'
    mpimg.imsave(save_name, (output_awb1 * 255).astype('uint8'))

print('d')

