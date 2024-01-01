import matplotlib.pyplot as plt
import numpy as np

img = plt.imread('/dataset/colorconstancy/set1_all1/Canon1DsMkIII_0004_F_AS.png')
gt = plt.imread('/dataset/colorconstancy/set1_all1/Canon1DsMkIII_0004_G_AS.png')

img_avg = np.average(img,axis=(0,1))
img_std = np.std(img,axis=(0,1))

gt_avg = np.average(gt,axis=(0,1))
gt_std = np.std(gt,axis=(0,1))

img_new,gt_new = np.zeros((749,1124,3)),np.zeros((749,1124,3))

for i in range(3):
    img_new[:,:,i] = (img[:,:,i]-img_avg[i])/img_std[i]
    gt_new[:, :, i] = img_new[:,:,i] * gt_std[i] + gt_avg[i]


plt.imshow(img)
plt.show()

plt.imshow(gt)
plt.show()

plt.imshow(img_new)
plt.show()

plt.imshow(gt_new)
plt.show()


print('d')


"数据更新"
    # camera_list = ["Canon1DsMkIII", "Canon600D", "FujifilmXM1", "NikonD40", "NikonD5200", "OlympusEPL6", "PanasonicGX1",
    #                "SamsungNX2000", "SonyA57","IMG"]
    #
    # data = np.load('/home/lcx/deep_final/deep_final/folds/train_all_5000_.npy',allow_pickle=True).item()
    # input = data['input']
    # s = data['s']
    # t = data['f']
    # awb = data['awb']
    # dir = '/dataset/colorconstancy/set1_all1/'
    #
    # name_input,s_input,t_input,awb_input,CAM = [],[],[],[],[]
    # for i in range(len(input)):
    #     name_input.append(dir+os.path.split(input[i])[1])
    #     s_input.append(dir+os.path.split(s[i])[1])
    #     t_input.append(dir+os.path.split(t[i])[1])
    #     awb_input.append(dir+os.path.split(awb[i])[1])
    #     "CAM"
    #     cam = os.path.split(input[i])[1].split("_")[0]
    #
    #     if cam in camera_list:
    #         CAM.append(camera_list.index(cam))
    #     else:
    #         CAM.append(10)
    #
    #
    #
    # data_new = {}
    # data_new['name'] = np.array(name_input)
    # data_new['s'] = np.array(s_input)
    # data_new['t'] = np.array(t_input)
    # data_new['gt'] = np.array(awb_input)
    # data_new['ct'] = data['label_cc']
    # data_new['name_list_cc'] = data['name_list_cc']
    # data_new['cam'] = np.array(CAM)
    #
    # np.save('/home/lcx/deep_final/deep_final/folds/train_all_5000_1_.npy',data_new)
    # print('d')