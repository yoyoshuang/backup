#coding:utf-8

import numpy as np
import os
# import imtool
import shutil
# import cv2
from PIL import Image
from pylab import *
from sklearn.cluster import KMeans
from scipy.misc import imsave
from scipy.ndimage import morphology,measurements
# import matplotlib.pyplot as plt
from skimage import draw 

imgname = './fapiao/009.jpeg'

img = np.array(Image.open(imgname))

# print(img.shape)

#去背景，kmeans，得到无背景图imgnoBG
img = img/255.0
w,h,d = tuple(img.shape)
img2D = img.reshape((w*h),d)
kmeans = KMeans(n_clusters = 2,random_state = 0).fit(img2D)
labels2D = kmeans.predict(img2D)

labels = labels2D.reshape(w,h)
num1 = sum(labels)
 #labels=1 的像素点占多数，labels=1的像素点为背景点
if num1>w*h/2: 
	maskbg = 1-labels
else:
	maskbg = labels.copy()
imgcopy = img.copy()
maskbg = (maskbg==0)
imgcopy[maskbg] = 0
imgnoBG = imgcopy

# figure()
# imshow(imgnoBG)
# show()

# 在去背景图的基础上提取红色信息，包括表格和红章
imgr = imgnoBG[:,:,0]
imgg = imgnoBG[:,:,1]
imgb = imgnoBG[:,:,2]

maskR1 = imgr>imgg
maskR2 = imgr>imgb
maskR = maskR1 * maskR2

imgcopy = imgnoBG.copy()
imgcopy[maskR] = 0
imgR = imgnoBG-imgcopy

maskB1 = imgb>imgr
maskB2 = imgb>imgg
maskB = maskB1*maskB2

imgcopy2 = imgnoBG.copy()
imgcopy2[maskB] = 0
imgB = imgnoBG-imgcopy2


# figure()
# imshow(imgR)
# show()

# imsave("fapiaoR.png",imgR)
# 在红色图上提取红章
imgRr = imgR[:,:,0]
imgRg = imgR[:,:,1]
imgRb = imgR[:,:,2]

maskstamp = imgRr>(imgRg+imgRb)*0.8
imgcopy = imgR.copy()
imgcopy[maskstamp] = 0
img_stamp = imgR-imgcopy
thirdw = int(w/3)
img_stamp_half = img_stamp[0:thirdw,:,:]
# print(img_stamp_half.shape)

# figure()
# imshow(img_stamp_half)
# show()

# imsave("stamp.png",img_stamp)

# 对红章图进行连通域标记，选择面积第二大的区域为红章区域
bn_stamp = np.zeros((thirdw,h))
maskstamp_third = maskstamp[0:thirdw,:]
bn_stamp[maskstamp_third] = 1

# bn_stamp = np.ones((w,h))
# bn_stamp[maskstamp] = 0
# gray()
# figure()
# imshow(bn_stamp)
# show()
# stamp_open = morphology.binary_erosion(bn_stamp,ones((2,2)),iterations = 2)
# stamp_open = morphology.binary_dilation(stamp_open,ones((2,2)),iterations = 1)
labels_open,nbr  = measurements.label(bn_stamp)
# count = zeros(nbr)
# print(nbr)
# imsave("label.png",labels_open)
# gray()
# figure()
# imshow(stamp_open)
# show()
count = zeros(nbr)
for i in range(nbr):
	count[i] = np.sum(labels_open==i)
	# print(count[i])
index = np.argsort(-count)[1]
# print(index)

maskstamponly = (labels_open==index)
# print(a.shape)
stamp_only = zeros((thirdw,h))
stamp_only[maskstamponly] = 1

# gray()
# figure()
# imshow(stamp_only)
# show()

#计算红章中心点坐标，计算红章宽高
stamp_points = np.where(stamp_only==1)
# print(points)
stamp_x = np.average(stamp_points[0])
stamp_y = np.average(stamp_points[1])

stamp_h = np.max(stamp_points[0])-np.min(stamp_points[0])
stamp_w = np.max(stamp_points[1])-np.min(stamp_points[1])
# print(stamp_x,stamp_y,stamp_w,stamp_h)

# 在原图上绘制红章中心点显示
# rr, cc=draw.circle(int(stamp_x),int(stamp_y),5)
# draw.set_color(img,[rr, cc],[0,255,0])
# figure()
# imshow(img)
# show()


#按比例获取截图区域，右下（金额），右上（发票号，日前），左上（号）三个区域
ratio_right = 1.15
ratio_right_r = 7
start_rt =int(stamp_w/ratio_right+np.max(stamp_points[1])) #列
# end_rt = int(np.max(stamp_points[0])+5) #行
end_rt = int(stamp_h/ratio_right_r+np.max(stamp_points[0])) 
croprighttop = img[0:end_rt,start_rt:h,:] # [行，列，：]

figure()
imshow(croprighttop)
show()

ratio_right2 = 0.28#0.3
start_rb_r = int(w/2)
end_rb_r = int(stamp_h/ratio_right2+np.max(stamp_points[0]))
start_rb_c = int(stamp_w/2+np.max(stamp_points[1]))
croprightbotm = img[start_rb_r:end_rb_r,start_rb_c:h,:]
figure()
imshow(croprightbotm)
show()

end_lf_c = int(np.min(stamp_points[1])-(stamp_w/1.15))
end_lf_r = int(stamp_x)
cropleft = imgnoBG[0:end_lf_r,0:end_lf_c,:]

figure()
imshow(cropleft)
show()












