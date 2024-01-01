import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import random
import pandas as pd
import math
import cv2
import os

def Error_from_txt(txt,cam): # 从txt文件建立图
    name, delta,mse,mae = [],[],[],[]
    for line in txt:
        print(line)
        if line != '\n':
            line = line.strip() # 去除首尾多余字符
            line = line.split() # 按空格分割
            nn = line[0].split(':')[1]
            print(nn)
            # if nn.split("_")[0] == cam:
            name.append(nn)
            delta.append(float(line[2].split(':')[1]))
            mse.append(float(line[3].split(':')[1]))
            mae.append(float(line[3].split(':')[1]))

    name = np.array(name)
    delta = np.array(delta)
    mse = np.array(mse)
    mae = np.array(mae)
    return name,delta,mse,mae

"Canon1DsMkIII"
error_wb1 = open("/home/lcx/artflow/error/WB_sRGB_Deltae.txt", "r", encoding="utf-8")
name_wb_can,delta_wb_can,mse_wb_can,mae_wb_can= Error_from_txt(error_wb1,'Canon1DsMkIII')
# error_wb2 = open("/home/lcx/artflow/error/WB_sRGB_Deltae.txt", "r", encoding="utf-8")
# name_wb_sam,delta_wb_sam,mse_wb_sam,mae_wb_sam= Error_from_txt(error_wb2,'SamsungNX2000')
# delta_wb = np.concatenate((delta_wb_can[:,np.newaxis],delta_wb_sam[:,np.newaxis]),axis=1)

error_unt1 = open("/home/lcx/artflow/error/UNET_R_Deltae.txt", "r", encoding="utf-8")
name_unt_can,delta_unt_can,mse_unt_can,mae_unt_can= Error_from_txt(error_unt1,'Canon1DsMkIII')
# error_unt2 = open("/home/lcx/artflow/error/UNET_R_Deltae.txt", "r", encoding="utf-8")
# name_unt_sam,delta_unt_sam,mse_unt_sam,mae_unt_sam= Error_from_txt(error_unt2,'SamsungNX2000')
# delta_unt = np.concatenate((delta_unt_can[:,np.newaxis],delta_unt_sam[:,np.newaxis]),axis=1)

error_glow1 = open("/home/lcx/artflow/error/GLOW_WB_RE_Deltae.txt", "r", encoding="utf-8")
name_glow_can,delta_glow_can,mse_glow_can,mae_glow_can= Error_from_txt(error_glow1,'Canon1DsMkIII')
# error_glow2 = open("/home/lcx/artflow/error/GLOW_WB_RE_Deltae.txt", "r", encoding="utf-8")
# name_glow_sam,delta_glow_sam,mse_glow_sam,mae_glow_sam= Error_from_txt(error_glow2,'SamsungNX2000')
# delta_glow = np.concatenate((delta_glow_can[:,np.newaxis],delta_glow_sam[:,np.newaxis]),axis=1)

# delta = np.concatenate((delta_wb,delta_unt,delta_glow),axis=1)

""

souce_dir = '/dataset/colorconstancy/NUS_CAM2/'
pos = '.jpg'
for i in range(101,151):
    pair_dir = souce_dir + str(i) + '/'

    k = 0
    count = 0

    err_wb, err_unt, err_glow = {},{},{}

    CC = ['28','38','55','65','75']
    for ii in range(5):
        err_wb[CC[ii]] = []
        err_unt[CC[ii]] = []
        err_glow[CC[ii]] = []

    for fn in os.listdir(pair_dir):
        if fn.lower().endswith(pos):
            print(fn)
            "相同色温，不同相机分一行"
            ct = fn.split("_")[-2]
            "三个算法的error"
            fn_ = fn.split('.')[0]

            if not(ct=='G'):
                error_wb = delta_wb_can[list(name_wb_can).index(fn_)]
                error_unt = delta_unt_can[list(name_unt_can).index(fn_)]
                error_glow = delta_glow_can[list(name_glow_can).index(fn_)]
                err_wb[ct].append(error_wb)
                err_unt[ct].append(error_unt)
                err_glow[ct].append(error_glow)




    print('d')

# ['OlympusEPL6','Canon600D','SonyA57','FujifilmXM1',
#  'PanasonicGX1','SamsungNX2000','NikonD5200','Canon1DsMkIII']