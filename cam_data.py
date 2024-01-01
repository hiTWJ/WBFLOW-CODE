import numpy as np
import os

"NikonD40"
"9:IMG,10:8D5U, 0:Canon1DsMKIII, 1: Caonon600D, 2:FujifilmXM1, 3:NikonD40,4:NikonD5200 "
camera_list = ["Canon1DsMkIII", "Canon600D", "FujifilmXM1","NikonD40","NikonD5200", "OlympusEPL6", "PanasonicGX1",
               "SamsungNX2000", "SonyA57","IMG","8D5U"]

data = np.load('/home/lcx/deep_final/deep_final/folds/train_PanasonicGX1_ct_5.npy',allow_pickle=True).item()

input_dir = data['input']
input_s_dir  = data['s']
input_t_dir  = data['t']
gt_dir = data['awb']

"1.改Input,s,t,awb的路径"
"2.加cam label"

new_dir = '/dataset/colorconstancy/set1_all1/'

data_new = {}
# data_new['ct'] = []
data_new['cam'] = []
data_new['name'] = []
data_new['name_s'] = []
data_new['name_t'] = []
data_new['gt'] = []
# data_new['scene'] = []
k=1
for i in range(input_dir.size):
    "1.改Input,s,t,awb的路径"
    ipt = input_dir[i]
    ipt_ = os.path.join(new_dir,os.path.split(ipt)[1])
    ips = input_s_dir[i]
    ipts_ = os.path.join(new_dir, os.path.split(ips)[1])
    iptt = input_t_dir[i]
    iptt_ = os.path.join(new_dir, os.path.split(iptt)[1])
    gt = gt_dir[i]
    gt_ = os.path.join(new_dir,os.path.split(gt)[1])
    "2.加cam label"
    cam_nam = os.path.split(ipt_)[1].split("_")[0]
    if cam_nam in camera_list:
        cam_ = camera_list.index(cam_nam)
    else:
        cam_ = 10
    "文件都存在"
    if os.path.exists(ipt_) and os.path.exists(ipts_) and os.path.exists(iptt_) and os.path.exists(gt_):
        print(k)
        k = k+1
        data_new['name'].append(ipt_)
        data_new['name_s'].append(ipts_)
        data_new['name_t'].append(iptt_)
        data_new['gt'].append(gt_)
        data_new['cam'].append(cam_)


data_new['cam'] = np.array(data_new['cam'])
data_new['name'] = np.array(data_new['name'])
data_new['name_s'] = np.array(data_new['name_s'])
data_new['name_t'] = np.array(data_new['name_t'])
data_new['gt'] = np.array(data_new['gt'])

np.save(f'/home/lcx/deep_final/deep_final/folds/train_all_12000_cam.npy', data_new)

print('d')


# idx = [109,112,117,119,133,139,144]
idx = [113,114,115,116,118,120,121]
# data_can1d = np.load('/home/lcx/deep_final/deep_final/folds/val_Canon1DsMkIII_ct_5.npy',allow_pickle=True).item()
# data_can6d = np.load('/home/lcx/deep_final/deep_final/folds/val_Canon600D_ct_5.npy',allow_pickle=True).item()
# data_fuj = np.load('/home/lcx/deep_final/deep_final/folds/val_FujifilmXM1_ct_5.npy',allow_pickle=True).item()
# data_nik = np.load('/home/lcx/deep_final/deep_final/folds/val_NikonD5200_ct_5.npy',allow_pickle=True).item()
# data_oly = np.load('/home/lcx/deep_final/deep_final/folds/val_OlympusEPL6_ct_5.npy',allow_pickle=True).item()
# data_pan = np.load('/home/lcx/deep_final/deep_final/folds/val_PanasonicGX1_ct_5.npy',allow_pickle=True).item()
# data_sam = np.load('/home/lcx/deep_final/deep_final/folds/val_SamsungNX2000_ct_5.npy',allow_pickle=True).item()
# data_sony = np.load('/home/lcx/deep_final/deep_final/folds/val_SonyA57_ct_5.npy',allow_pickle=True).item()

for i in range(len(camera_list)):
    "建立新data"
    data_new = {}
    data_new['ct'] = []
    data_new['cam'] = []
    data_new['name'] = []
    data_new['gt'] = []
    data_new['scene'] = []

    cam = camera_list[i]
    print(cam)
    data = np.load('/home/lcx/deep_final/deep_final/folds/val_'+cam+'_ct_5.npy',allow_pickle=True).item()
    scene = data['scene']
    for j in range(len(idx)):
        data_idx = np.where(scene==idx[j])
        data_new['ct'].extend(list(data['ct'][data_idx]))
        data_new['cam'].extend(list(data['cam'][data_idx]))
        data_new['name'].extend(list(data['name'][data_idx]))
        data_new['gt'].extend(list(data['gt'][data_idx]))
        data_new['scene'].extend(list(data['scene'][data_idx]))

    data_new['ct'] = np.array(data_new['ct'])
    data_new['cam'] = np.array(data_new['cam'])
    data_new['name'] = np.array(data_new['name'])
    data_new['gt'] = np.array(data_new['gt'])
    data_new['scene'] = np.array(data_new['scene'])
    np.save(f'/home/lcx/deep_final/deep_final/folds/val_'+cam+'_sample1.npy', data_new)

print('d')



souce_dir = '/dataset/colorconstancy/NUS_CAM1/'
pos = '.jpg'
list_ = [64,76,100]
"101"


# for ii in range(0,len(camera_list)):
data = {}
CT, CAM, NAME, GT, SC = [], [], [], [], []
for i in range(1,101):
    pair_dir = souce_dir + str(i) + '/'

    k = 0
    count = 0
    for fn in os.listdir(pair_dir):
        if fn.lower().endswith(pos) and not (fn.split("_")[-2] == 'G'):
        # if fn.lower().endswith(pos) and not(fn.split("_")[-2] == 'G') and fn.split("_")[0] == camera_list[ii]:
            "WB图像"
            print(pair_dir+fn)
            sp = fn.split("_")
            ct = sp[-2]
            cam = sp[0]
            # if cam == 'OlympusEPL6':
            #     count = count+1
            name = pair_dir + fn

            "GT"
            sp[-2] = 'G'
            gt_dir = pair_dir + cam
            for p in range(1,len(sp)):
                gt_dir = gt_dir+ "_"  + sp[p]

            if os.path.exists(gt_dir) and os.path.exists(pair_dir+fn):
                CT.append(int(ct))
                CAM.append(camera_list.index(cam)+1)
                NAME.append(pair_dir + fn)
                GT.append(gt_dir)
                SC.append(i)
    # print(count)



data['ct'] = np.array(CT)
data['cam'] = np.array(CAM)
data['name'] = np.array(NAME)
data['gt'] = np.array(GT)
data['scene'] = np.array(SC)
# np.save(f'/home/lcx/deep_final/deep_final/folds/train_{camera_list[ii]}_ct_5.npy',data)
np.save(f'/home/lcx/deep_final/deep_final/folds/train_cam8_ct_5.npy',data)

print('d')
