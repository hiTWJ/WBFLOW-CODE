import argparse
import ast
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import torch
from torch import nn
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader, random_split
from utilities import utils as utls
from evaluation.evaluate_cc import evaluate_cc


from utilities.crop_val import MultiEvalModule

import warnings
warnings.filterwarnings("ignore")
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

from arch import deep_wb_single_task
from utilities.dataset_cam import BasicDataset
from evaluation.calc_deltaE2000 import calc_deltaE2000



def load_state(net,load):
    " 加载stage1 , map_location=cuda:0"
    net.load_state_dict(
        torch.load(os.path.join(load)))
    return net

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf_logger", type=ast.literal_eval, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--ckpt", default='/home/chunxiao/FACT/fact_out/PACS_ResNet50/logs/2022-08-09-23-58-40/', help="The directory to models")
    parser.add_argument("--input_dir", default=None, help="The directory of dataset lists")
    parser.add_argument("--output_dir", default='/home/chunxiao/FACT/fact_out/',
                        help="The directory to save logs and models")
    parser.add_argument('-dan', '--data-name', dest='data_name', default='cube_all',
                        help='Training camera')

    args = parser.parse_args()
    return args


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
        self.type  = 'net_awb'
        self.net = deep_wb_single_task.deepWBnet().to(device)
        self.net = load_state(self.net,'/home/chunxiao/deep_final/PyTorch/models/'+self.type +'.pth')

        # dataloaders
        "Cube: 相同场景，不同色温测试图像"
        self.pz = 256
        self.valt_c1 = BasicDataset(patch_size=self.pz, type='val', ct=2, camtype='Canon1DsMkIII')
        self.val_loader_c1 = DataLoader(self.valt_c1, batch_size=self.bs, shuffle=False, num_workers=4, pin_memory=True)
        self.valt_c6 = BasicDataset(patch_size=self.pz, type='val', ct=2, camtype='Canon600D')
        self.val_loader_c6 = DataLoader(self.valt_c6, batch_size=self.bs, shuffle=False, num_workers=4, pin_memory=True)
        self.valt_f = BasicDataset(patch_size=self.pz, type='val', ct=2, camtype='FujifilmXM1')
        self.val_loader_f = DataLoader(self.valt_f, batch_size=self.bs, shuffle=False, num_workers=4, pin_memory=True)
        # self.valt_n4 = BasicDataset(patch_size=self.pz, type='train', ct=2, camtype='NikonD40')
        # self.val_loader_n4 = DataLoader(self.valt_n4, batch_size=self.bs, shuffle=False, num_workers=4, pin_memory=True)
        self.valt_n = BasicDataset(patch_size=self.pz, type='val', ct=2, camtype='NikonD5200')
        self.val_loader_n = DataLoader(self.valt_n, batch_size=self.bs, shuffle=False, num_workers=4, pin_memory=True)
        self.valt_o = BasicDataset(patch_size=self.pz, type='val', ct=2, camtype='OlympusEPL6')
        self.val_loader_o = DataLoader(self.valt_o, batch_size=self.bs, shuffle=False, num_workers=4, pin_memory=True)
        self.valt_p = BasicDataset(patch_size=self.pz, type='val', ct=2, camtype='PanasonicGX1')
        self.val_loader_p = DataLoader(self.valt_p, batch_size=self.bs, shuffle=False, num_workers=4, pin_memory=True)
        self.valt_sn = BasicDataset(patch_size=self.pz, type='val', ct=2, camtype='SamsungNX2000')
        self.val_loader_sn = DataLoader(self.valt_sn, batch_size=self.bs, shuffle=False, num_workers=4, pin_memory=True)
        self.valt_sa = BasicDataset(patch_size=self.pz, type='val', ct=2, camtype='SonyA57')
        self.val_loader_sa = DataLoader(self.valt_sa, batch_size=self.bs, shuffle=False, num_workers=4, pin_memory=True)

        "txt"
        self.error_txt = '/home/lcx/deep_final/deep_final/PyTorch/ERROR/' + self.type+'_Deltae.txt'

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
        np.save('/home/lcx/deep_final/deep_final/PyTorch/ERROR/' + self.type + '_'+ type, SC)



    def do_eval(self, loader,device):
        D,MS,MA = [],[],[]

        for it, (batch) in enumerate(loader):
            "image"
            gt = batch['gt-AWB'].to(device=device, dtype=torch.float32)
            imgs = batch['image'].to(device=device, dtype=torch.float32)
            "label"
            name = batch['dir'].to(device=device, dtype=torch.float32)
            ct = batch['ct'][0].to(device=device, dtype=torch.float32)
            cam = batch['cam'].to(device=device, dtype=torch.float32)

            "只保留名称"
            name_ = os.path.split(name)[1]

            out_img = self.net(imgs)

            ""
            gt_ = gt.cpu().detach().numpy()[0].transpose(1,2,0)
            # pre_img  =pre_img.cpu().detach().numpy()[0].transpose(1,2,0)
            # pre_img[78:, 102:, :] = 0
            # pre_img1 = pre_img1.cpu().detach().numpy()[0].transpose(1, 2, 0)
            out_img_ =out_img.cpu().detach().numpy()[0].transpose(1,2,0)
            # tf = transforms.Compose([
            #     transforms.ToPILImage(),
            #     transforms.ToTensor()
            # ])

            # output_awb = tf(torch.squeeze(pre_img.cpu()))
            # output_awb = output_awb.squeeze().cpu().numpy()
            # output_awb = output_awb.transpose((1, 2, 0))
            # # output_awb[78:, 102:, :] = 0
            # # plt.imshow(output_awb)
            # # plt.show()
            #
            # m_awb = utls.get_mapping_func(imgs_, output_awb)
            # output_awb_ = utls.outOfGamutClipping(utls.apply_mapping_func(imgs_, m_awb))

            deltaE00, MSE, MAE = evaluate_cc(out_img_ * 255, gt_ * 255, 0, opt=3)

            "存txt"
            error_save = open(self.error_txt, mode='a')
            error_save.write(
                '\n' + 'Name:' + str(name_) + ' Cam:' +self.cam_list[cam] + ' CT:'+str(ct)+ '  DELTA E:' + str(deltaE00))
            # 关闭文件
            error_save.close()
            D.append(deltaE00)
            MS.append(MSE[0])
            MA.append(MAE)

        D = np.array(D)
        MS = np.array(MS)
        MA = np.array(MA)

        return D,MS,MA

    def do_testing(self):
        self.net.eval()

        with torch.no_grad():
            print('1. Canon1DsMkIII:')
            self.save_e(self.val_loader_c1, type='Canon1DsMkIII')
            print('2. Canon600D:')
            self.save_e(self.val_loader_c6, type='Canon600D')
            print('3. FujifilmXM1:')
            self.save_e(self.val_loader_f, type='FujifilmXM1')
            print('4. NikonD5200:')
            self.save_e(self.val_loader_c1, type='NikonD5200')
            print('5. OlympusEPL6:')
            self.save_e(self.val_loader_o, type='OlympusEPL6')
            print('6. PanasonicGX1:')
            self.save_e(self.val_loader_p, type='PanasonicGX1')
            print('7. SamsungNX2000:')
            self.save_e(self.val_loader_sn, type='SamsungNX2000')
            print('8. SonyA57:')
            self.save_e(self.valt_sa, type='SonyA57')




def main():
    args = get_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    evaluator = Evaluator(args, device)
    evaluator.do_testing()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
