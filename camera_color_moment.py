import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

"计算图像颜色直方图+颜色矩"


def histogram(self, image, bins):
    # 使用提供的每个通道的 bin 数量，从图像的遮罩区域中提取 3D 颜色直方图
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    # 返回直方图
    return hist

def color_moments(img):
    # Split the channels - h,s,v
    h, s, v = cv2.split(img)
    # Initialize the color feature
    color_feature = []
    # N = h.shape[0] * h.shape[1]
    # The first central moment - average
    h_mean = np.mean(h)  # np.sum(h)/float(N)
    s_mean = np.mean(s)  # np.sum(s)/float(N)
    v_mean = np.mean(v)  # np.sum(v)/float(N)
    avg_mean = (h_mean + s_mean + v_mean) / 3
    # color_feature.extend(avg_mean)

    # The second central moment - standard deviation
    h_std = np.std(h)  # np.sqrt(np.mean(abs(h - h.mean())**2))
    s_std = np.std(s)  # np.sqrt(np.mean(abs(s - s.mean())**2))
    v_std = np.std(v)  # np.sqrt(np.mean(abs(v - v.mean())**2))
    avg_std = (h_std+s_std+v_std) / 3
    # color_feature.extend(avg_std)

    # The third central moment - the third root of the skewness
    # h_skewness = np.mean(abs(h - h.mean())**3)
    # s_skewness = np.mean(abs(s - s.mean())**3)
    # v_skewness = np.mean(abs(v - v.mean())**3)
    #
    # h_thirdMoment = h_skewness**(1./3)
    # s_thirdMoment = s_skewness**(1./3)
    # v_thirdMoment = v_skewness**(1./3)
    # avg_thirdMoment= (h_thirdMoment + s_thirdMoment + v_thirdMoment) / 3
    # color_feature.extend(avg_thirdMoment)
    # return np.array([avg_mean,avg_std,avg_thirdMoment])
    return [h_mean, h_std]

## "NikonD40",

# camera_list = ["Canon1DsMkIII", "Canon600D", "FujifilmXM1", "NikonD5200", "OlympusEPL6", "PanasonicGX1",
#                "SamsungNX2000", "SonyA57"]

camera_list = ["NikonD5200", "FujifilmXM1", "Canon600D","Canon1DsMkIII","PanasonicGX1","SamsungNX2000","OlympusEPL6", "SonyA57"]

souce_dir = '/dataset/colorconstancy/NUS_CAM1/'
pos = '.jpg'
list_ = [64,76,100]
"101"
"sample: 53"
CC = {}
for cam_ in camera_list:
    CC[cam_] = []

for i in range(1,101):
    pair_dir = souce_dir + str(i) + '/'
    print(pair_dir)

    img_dct_ = np.zeros((8,3,100))
    k = 0
    # CC_ = {}

    for fn in os.listdir(pair_dir):
        if fn.lower().endswith(pos):
            "WB图像"
            kk = fn.split("_")
            if fn.split("_")[-2] == 'G'and not(fn.split("_")[0] == '.'):
                img_dir = os.path.join(pair_dir,fn)
                "读取数据"
                "相机"
                cam = fn.split('_')[0]
                print(img_dir)
                img = cv2.imread(img_dir)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                "YCbCr"
                img = cv2.resize(img, (256, 256))
                # plt.imshow(img)
                # plt.show()
                "颜色矩"
                color_feature = color_moments(img/255)
                CC[cam].append(color_feature)

cam_mean,cam_std = [],[]
for cam_ in camera_list:
    # CC[cam_] = np.array(CC[cam_])
    cam_mean.append(list(np.array(CC[cam_])[:98,0]))
    cam_std.append(list(np.array(CC[cam_])[:98,1]))

cam_mean,cam_std = np.array(cam_mean), np.array(cam_std)
save_name = souce_dir + 'camera_color_moments_list.npy'
np.save(save_name, np.array(CC))

print('DONE!!')


# count_r = cv2.calcHist(img, [0], None, [256], [0.0, 255.0])/(256*256)
# count_g = cv2.calcHist(img, [1], None, [256], [0.0, 255.0])/(256*256)
# count_b = cv2.calcHist(img, [2], None, [256], [0.0, 255.0])/(256*256)
# count = np.concatenate((count_r,count_g,count_b),axis=1)
# CC[cam] = color_feature + CC[cam]
# x = np.array(range(256))

# plt.bar(range(256),count_r[:,0],color='r')
# plt.bar(range(256),count_g[:,0],color='g')
# plt.bar(range(256), count_b[:, 0], color='b')
# plt.show()
# img_dct[cam] = img_dct[cam] +count

#
# for cam2_ in camera_list:
#     CC[cam2_] = CC[cam2_] /100


# CC = np.array(CC)
# CC = np.average(CC,axis=0)


