import os
import cv2
import math
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
from shutil import copyfile
import time

   
# 判断左右
def location_lr(img, path, filename):
    h,w,k = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    standard_path =  path + 'standard/'
    if not os.path.exists(standard_path):
        os.makedirs(standard_path)
    error_path = path + 'error/'
    if not os.path.exists(error_path):
        os.makedirs(error_path)
    right_path = path + 'right/'
    if not os.path.exists(right_path):
        os.makedirs(right_path)
    right_path1 = right_path + '1/'
    if not os.path.exists(right_path1):
        os.makedirs(right_path1)
    right_path2 = right_path + '2/'
    if not os.path.exists(right_path2):
        os.makedirs(right_path2)
    left_path = path + 'left/'
    if not os.path.exists(left_path):
        os.makedirs(left_path)
    left_path1 = left_path + '1/'
    if not os.path.exists(left_path1):
        os.makedirs(left_path1)
    left_path2 = left_path + '2/'
    if not os.path.exists(left_path2):
        os.makedirs(left_path2)
    
    left_dis = 0
    right_dis = 0
    for i in range(20,600,1):
        #print(i,int(np.var(img[int(0.25*h):int(0.7*h),i:i+5])-np.var(img[int(0.25*h):int(0.7*h),int(0.01*w)-20:int(0.01*w)-15])))
        if int(np.var(img[50:int(0.6*h),i:i+5])-np.var(img[50:int(0.6*h),15:20])) > 100:#40
            #print(i)
            left_dis = i
            break

    for i in range(w-20,w-600,-1):
        #print(i,int(np.var(img[int(0.25*h):int(0.7*h),i-5:i])-np.var(img[int(0.25*h):int(0.7*h),int(0.99*w)+15:int(0.99*w)+20])))
        if int(np.var(img[50:int(0.6*h),i-5:i])-np.var(img[50:int(0.6*h),w-20:w-15])) > 100:#80
            #print(w - i)
            right_dis = w - i
            break

    if left_dis == 0:
        for i in range(20,600,1):
            #print(i,int(np.var(img[int(0.25*h):int(0.7*h),i:i+5])-np.var(img[int(0.25*h):int(0.7*h),int(0.01*w)-20:int(0.01*w)-15])))
            if abs(int(np.var(img[50:int(0.6*h),i:i+5])-np.var(img[50:int(0.6*h),15:20]))) > 40:#40
                #print(i)
                left_dis = i
                break
    if right_dis == 0:
        for i in range(w-20,w-600,-1):
            #print(i,int(np.var(img[int(0.25*h):int(0.7*h),i-5:i])-np.var(img[int(0.25*h):int(0.7*h),int(0.99*w)+15:int(0.99*w)+20])))
            if abs(int(np.var(img[50:int(0.6*h),i-5:i])-np.var(img[50:int(0.6*h),w-20:w-15]))) > 40:#80
                #print(w - i)
                right_dis = w - i
                break    
    print(left_dis, right_dis)
    
    if abs(left_dis - right_dis) > 0.1*w:
        print('识别错误',filename)
        copyfile(path + filename, error_path + filename)
        return 0
    
    if left_dis - right_dis > 0.003*w:
        print('向右偏',filename)
        if left_dis - right_dis > 0.015*w:
            copyfile(path + filename, right_path2 + filename)
        else:
            copyfile(path + filename, right_path1 + filename)
    elif right_dis - left_dis > 0.003*w:
        print('向左偏',filename)
        if right_dis - left_dis > 0.015*w:
            copyfile(path + filename, left_path2 + filename)
        else:
            copyfile(path + filename, left_path1 + filename)
    else:
        print('左右不偏',filename)
        copyfile(path + filename, standard_path + filename)
        

#判断上下    
def location_ud(img, path, filename):
    h,w,k = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    standard_path =  path + 'standard/'
    if not os.path.exists(standard_path):
        os.makedirs(standard_path)
    error_path =  path + 'error/'
    if not os.path.exists(error_path):
        os.makedirs(error_path)
    up_path =  path + 'up/'
    if not os.path.exists(up_path):
        os.makedirs(up_path)
    up_path1 =  up_path + '1/'
    if not os.path.exists(up_path1):
        os.makedirs(up_path1)
    up_path2 =  up_path + '2/'
    if not os.path.exists(up_path2):
        os.makedirs(up_path2)
    down_path =  path + 'down/'
    if not os.path.exists(down_path):
        os.makedirs(down_path)
    down_path1 =  down_path + '1/'
    if not os.path.exists(down_path1):
        os.makedirs(down_path1)
    down_path2 =  down_path + '2/'
    if not os.path.exists(down_path2):
        os.makedirs(down_path2)
    
    down_dis = 0
    up_dis = 0
    for i in range(10,400,1):
        if int(np.var(img[i:i+5,50:int(0.6*h)])-np.var(img[5:10,50:int(0.6*h)])) > 40:#40
            #print(i)
            up_dis = i
            break
    for i in range(h-10,h-400,-1):
        if int(np.var(img[i-5:i,50:int(0.6*h)])-np.var(img[h-10:h-5,50:int(0.6*h)])) > 40:#80
            #print(w - i)
            down_dis = h - i
            break
        
        
    if up_dis == 0:
        for i in range(10,400,1):
            if abs(int(np.var(img[i:i+5,50:int(0.6*h)])-np.var(img[5:10,50:int(0.6*h)]))) > 20:#40

                up_dis = i
                break
    if down_dis == 0:
        for i in range(h-10,h-400,-1):
            if abs(int(np.var(img[i-5:i,50:int(0.6*h)])-np.var(img[h-10:h-5,50:int(0.6*h)]))) > 20:#80
                down_dis = h - i
                break  
    print(down_dis, up_dis)
    
    if abs(down_dis - up_dis) > 0.1*h:
        print('识别错误',filename)
        copyfile(path + filename, error_path + filename)
    
    if down_dis - up_dis > 0.003*h:
        print('向上偏',filename)
        if down_dis - up_dis > 0.015*h:
            copyfile(path + filename, up_path2 + filename)
        else:
            copyfile(path + filename, up_path1 + filename)
    elif up_dis - down_dis > 0.003*h:
        print('向下偏',filename)
        if up_dis - down_dis > 0.015*h:
            copyfile(path + filename, down_path2 + filename)
        else:
            copyfile(path + filename, down_path1 + filename)
    else:
        print('上下不偏',filename)
        copyfile(path + filename, standard_path + filename)

    
    
    
    
if __name__ == "__main__":
    time1 = time.time()
    root_path = r"D:/task-7/location/pat_pos/cut/"
    list_file = os.listdir(root_path)
    for file in list_file:
        if file[-3:] == 'jpg':
            data = np.fromfile(root_path + file, dtype=np.uint8)  
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            location_lr(img,root_path,file)
    time2 = time.time()
    #print('用时：',time2-time1)
    
    
    
    
    
    