#coding:utf-8

import numpy as np
import os
# import imtool
import shutil
import cv2
from PIL import Image,ImageDraw
from pylab import *
from sklearn.cluster import KMeans
from scipy.misc import imsave
from scipy.ndimage import morphology,measurements
# import matplotlib.pyplot as plt
from skimage import draw 
import matplotlib.pyplot as plt
import image_pross

imgname = './fapiao/001.jpg'


#放弃7图和2图,3图
img_org =  Image.open(imgname)
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
maskB_inv = np.logical_not(maskB)

# imgcopy2 = imgnoBG.copy()
imgcopy2 = np.zeros((w,h))
imgcopy2[maskB] = 1
imgcopy2[maskB_inv] = 0
# imgB = imgnoBG-imgcopy2
imgB  = imgcopy2

# imgcopy3 = imgnoBG.copy()
# maskB = np.logical_not(maskB)
# imgcopy3[maskB] = 0
# imgnoB = img-imgcopy3

# figure()
# imshow(imgB)
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
croprighttop = imgB[0:end_rt,start_rt:h] # [行，列，：]

# gray()
# figure()
# imshow(croprighttop)
# show()

ratio_right2 = 0.28#0.3
start_rb_r = int(w/2)
end_rb_r = int(stamp_h/ratio_right2+np.max(stamp_points[0]))
start_rb_c = int(stamp_w/2+np.max(stamp_points[1]))
croprightbotm = imgB[start_rb_r:end_rb_r,start_rb_c:h]
# figure()
# imshow(croprightbotm)
# show()

ratio_left = 1.15
ratio_left2 = 0.475
star_lf_c = int(np.min(stamp_points[1])-(stamp_w/ratio_left2))
end_lf_c = int(np.min(stamp_points[1])-(stamp_w/ratio_left))
end_lf_r = int(stamp_x)
cropleft = img[0:end_lf_r,star_lf_c:end_lf_c,:]

# figure()
# imshow(cropleft)
# show()

#在局部区域分割四要素

#左上提取发票号

# print(cropleft[5,6,0])

cropleft_lab = cv2.cvtColor(np.uint8(cropleft*255),cv2.COLOR_BGR2Lab)[:,:,0]
_ , cropleft_bn = cv2.threshold(cropleft_lab,120,255,cv2.THRESH_BINARY)

# croprightbotm_lab = cv2.cvtColor(np.uint8(croprightbotm*255),cv2.COLOR_BGR2Lab)[:,:,0]
# _ , croprightbotm_lab_bn = cv2.threshold(croprightbotm_lab,180,255,cv2.THRESH_BINARY)

# print(cropleft_lab)
# gray()
# figure()
# imshow(croprighttop_bn)
# show()
# print(cropleft_bn_c)

# 右上提取发票号，编号，日期
#行方向投影,取最后一行日期
# print(croprighttop[3,4,0])
# croprighttop_bn = 1- np.array(croprighttop)/255
index =  image_pross.projection(croprighttop,"row",8)
index = -np.sort(-index[0])
# print(index)
data_start_r = int((index[1]+index[2])/2)

crop_numbers = croprighttop[0:data_start_r,:]
# gray()
# figure()
# imshow(crop_numbers)
# show()
index = image_pross.projection(crop_numbers,"col",5)[0]
# print(index)
#取 index序列中距离最大的两个点求中间位置
diff_index = image_pross.diff_seq(index)
# print(diff_index)
indexmax = np.argmax(diff_index)
# print(indexmax)
# number1_end_c = int((index[15]+index[16])/2)
number1_end_c = int((index[indexmax]+index[indexmax-1])/2)
# print(index)

crop_numbers1 = croprighttop[0:data_start_r,number1_end_c:]
# gray()
# figure()
# imshow(crop_numbers1)
# show()
index = image_pross.projection(crop_numbers1,"row",5)
index = -np.sort(-index[0])
code2_end_r = int((index[1]+index[2])/2)
code2_start_r = int(index[3]-10)
crop_code2 = croprighttop[code2_start_r:code2_end_r,number1_end_c:]
# gray()
# figure()
# imshow(crop_code2)
# show()
crop_number2 = croprighttop[code2_end_r:data_start_r,number1_end_c:]
# gray()
# figure()
# imshow(crop_number2)
# show()

crop_data = croprighttop[data_start_r:,:]
index = image_pross.projection(crop_data,"col",5)[0]
data_start_c = int(index[0]-10)
crop_data = croprighttop[data_start_r:,data_start_c:]
# gray()
# figure()
# imshow(crop_data)
# show()

#右下提取不含税金额
index = image_pross.projection(croprightbotm,"row",5)[0]
mh = croprightbotm.shape[0]
money_start_r = int(index[0]-7)
money_end_r = np.min((int(index[1]+7),mh))
crop_money_tmp = croprightbotm[money_start_r:money_end_r,:]
# gray()
# figure()
# imshow(crop_money_tmp)
# show()
ratio_money_end_c = 0.53
index = image_pross.projection(crop_money_tmp,"col",5)[0]
money_start_c = int(index[0]-7)

diff_index = image_pross.diff_seq(index)
# print(diff_index)
indexmax = np.argmax(diff_index)
# print(indexmax)
# number1_end_c = int((index[15]+index[16])/2)
money_end_c = int((index[indexmax]+index[indexmax-1])/2)
# print(index)

# money_end_c = int(stamp_w/ratio_money_end_c+np.max(stamp_points[1]))-start_rt
crop_money = croprightbotm[money_start_r:money_end_r,money_start_c:money_end_c]
# gray()
# figure()
# imshow(crop_money)
# show()
# print(money_end_c)

#box = [左，上，右，下]
code1_box = [star_lf_c,0,end_lf_c,end_lf_r]
code2_box = [start_rt+number1_end_c,code2_start_r,h-1,code2_end_r]
number1_box = [start_rt,0,start_rt+number1_end_c,code2_end_r]
number2_box = [start_rt+number1_end_c,code2_end_r,h-1,data_start_r] 
data_box = [start_rt+data_start_c,data_start_r,h-1,end_rt]
money_box = [start_rb_c+money_start_c,start_rb_r+money_start_r,start_rb_c+money_end_c,start_rb_r+money_end_r]
# print(img.shape,data_box)

imgout = image_pross.draw_box(img_org,code1_box,"code1")
imgout = image_pross.draw_box(imgout,code2_box,"code2")
imgout = image_pross.draw_box(imgout,number1_box,"number1")
imgout = image_pross.draw_box(imgout,number2_box,"number2")
imgout = image_pross.draw_box(imgout,data_box,"date")
imgout = image_pross.draw_box(imgout,money_box,"money")

figure()
imshow(imgout)
show()
