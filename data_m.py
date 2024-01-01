import numpy as np
import os

camera_list = ["Canon1DsMkIII", "Canon600D", "FujifilmXM1", "NikonD5200", "OlympusEPL6", "PanasonicGX1",
               "SamsungNX2000", "SonyA57"]
souce_dir = '/dataset/colorconstancy/NUS_CAM1/'
pos = '.jpg'
# list_ = [64,76,100]
"101"


# for ii in range(0,len(camera_list)):


for cam_ in camera_list:
    data = {}
    CT, CAM, NAME, GT, SC = [], [], [], [], []
    print(cam_)
    for i in range(1,101):
        pair_dir = souce_dir + str(i) + '/'
        k = 0
        count = 0
        for fn in os.listdir(pair_dir):
            if fn.lower().endswith(pos) and fn.split("_")[0] == cam_ and not(fn.split("_")[-2] == 'G'):
                "WB图像"
                # print(pair_dir+fn)
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

                if os.path.exists(gt_dir):
                    CT.append(int(ct))
                    CAM.append(camera_list.index(cam))
                    NAME.append(pair_dir + fn)
                    GT.append(gt_dir)
                    SC.append(i)
                    k = k+1
        # print(i)
        # print(k)
        # print(count)



    data['ct'] = np.array(CT)
    data['cam'] = np.array(CAM)
    data['name'] = np.array(NAME)
    data['gt'] = np.array(GT)
    data['scene'] = np.array(SC)

    print(np.shape(CT)[0])
    # np.save(f'/home/lcx/deep_final/deep_final/folds/train_{camera_list[ii]}_ct_5.npy',data)
    np.save(f'/home/lcx/deep_final/deep_final/folds/train_'+cam_+'_ct_5.npy',data)

print('d')






