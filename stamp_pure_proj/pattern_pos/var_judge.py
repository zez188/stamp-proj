import os
import cv2
import math
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
from shutil import copyfile
import time


# 判断左右
# def location_lr(img, path, filename):
def location_lr(img):

    h,w,k = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    left_dis = 0
    right_dis = 0
    for i in range(20,600,1):
        #print(i,int(np.var(img[int(0.25*h):int(0.7*h),i:i+5])-np.var(img[int(0.25*h):int(0.7*h),int(0.01*w)-20:int(0.01*w)-15])))
        if int(np.var(img[int(0.25*h):int(0.7*h),i:i+5])-np.var(img[int(0.25*h):int(0.7*h),15:20])) > 80:#40
            #print(i)
            left_dis = i
            break
    for i in range(w-20,w-600,-1):
        #print(i,int(np.var(img[int(0.25*h):int(0.7*h),i-5:i])-np.var(img[int(0.25*h):int(0.7*h),int(0.99*w)+15:int(0.99*w)+20])))
        if int(np.var(img[int(0.25*h):int(0.7*h),i-5:i])-np.var(img[int(0.25*h):int(0.7*h),w-20:w-15])) > 80:#80
            #print(w - i)
            right_dis = w - i
            break

    if left_dis == 0:
        for i in range(20,600,1):
            #print(i,int(np.var(img[int(0.25*h):int(0.7*h),i:i+5])-np.var(img[int(0.25*h):int(0.7*h),int(0.01*w)-20:int(0.01*w)-15])))
            if abs(int(np.var(img[int(0.25*h):int(0.7*h),i:i+5])-np.var(img[int(0.25*h):int(0.7*h),15:20]))) > 20:#40
                #print(i)
                left_dis = i
                break
    if right_dis == 0:
        for i in range(w-20,w-600,-1):
            #print(i,int(np.var(img[int(0.25*h):int(0.7*h),i-5:i])-np.var(img[int(0.25*h):int(0.7*h),int(0.99*w)+15:int(0.99*w)+20])))
            if abs(int(np.var(img[int(0.25*h):int(0.7*h),i-5:i])-np.var(img[int(0.25*h):int(0.7*h),w-20:w-15]))) > 20:#80
                #print(w - i)
                right_dis = w - i
                break    
    print(left_dis, right_dis)
    if left_dis - right_dis > 20:
        # print('向右偏', filename)
        print('向右偏')
        lr_res = '向右偏'
        # copyfile(path + filename, right_path + filename)
    elif right_dis - left_dis > 20:
        # print('向左偏', filename)
        print('向左偏')
        lr_res = '向左偏'
        # copyfile(path + filename, left_path + filename)
    else:
        # print('左右不偏', filename)
        print('左右不偏')
        lr_res = None
    return lr_res

# 判断上下
# def location_ud(img, path, filename):
def location_ud(img):

    h,w,k = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # up_path =  path + 'up/'
    # if not os.path.exists(up_path):
    #     os.makedirs(up_path)
    # down_path =  path + 'down/'
    # if not os.path.exists(down_path):
    #     os.makedirs(down_path)
    up_dis = 0
    down_dis = 0
    for i in range(10,400,1):
        if int(np.var(img[i:i+5,int(0.25*w):int(0.7*w)])-np.var(img[5:10,int(0.25*w):int(0.7*w)])) > 80:#40
            #print(i)
            up_dis = i
            break
    for i in range(h-10,h-400,-1):
        if int(np.var(img[i-5:i,int(0.25*w):int(0.7*w)])-np.var(img[h-10:h-5,int(0.25*w):int(0.7*w), ])) > 80:#80
            #print(w - i)
            down_dis = h - i
            break
        
        
    if up_dis == 0:
        for i in range(10,400,1):
            if abs(int(np.var(img[i:i+5,int(0.25*w):int(0.7*w)])-np.var(img[5:10,int(0.25*w):int(0.7*w)]))) > 30:#40
                up_dis = i
                break
    if down_dis == 0:
        for i in range(h-10,h-400,-1):
            if abs(int(np.var(img[i-5:i, int(0.25*w):int(0.7*w)])-np.var(img[h-10:h-5, int(0.25*w):int(0.7*w), ]))) > 30:#80
                down_dis = h - i
                break  
    print(up_dis, down_dis)
    if up_dis - down_dis > 7:
        print('向下偏')
        ud_res = '向下偏'
        # copyfile(path + filename, down_path + filename)
    elif down_dis - up_dis > 7:
        # print('向上偏', filename)
        print('向上偏')
        ud_res = '向上偏'
        # copyfile(path + filename, up_path + filename)
    else:
        print('上下不偏')
        ud_res = None
    return ud_res
 
    
    
def judge_pat_pos(file_path):
    data = np.fromfile(file_path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    lr_res = location_lr(img)
    ud_res = location_ud(img)
    return lr_res, ud_res

if __name__ == "__main__":

    # time1 = time.time()
    # root_path = r"D:/task-end/cut/"
    # list_file = os.listdir(root_path)
    # for file in list_file:
    #     if file[-3:] == 'jpg':
    #         data = np.fromfile(root_path + file, dtype=np.uint8)
    #         img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    #         location_lr(img,root_path, file)
    # time2 = time.time()
    #print('用时：',time2-time1)

    pass
    
    
    
    
    
    