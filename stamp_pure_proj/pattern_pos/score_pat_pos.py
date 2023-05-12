import os
import cv2
import math
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
from shutil import copyfile
import time

   
# 判断左右
def location_lr(img, filename):
    h, w, k = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    left_dis = 0
    right_dis = 0

    score_pos_lr = 0

    for i in range(20, 600, 1):
        if int(np.var(img[50:int(0.6*h),i:i+5])-np.var(img[50:int(0.6*h),15:20])) > 100:#40
            left_dis = i
            break
    for i in range(w-20, w-600, -1):
        if int(np.var(img[50:int(0.6*h),i-5:i])-np.var(img[50:int(0.6*h),w-20:w-15])) > 100:#80
            right_dis = w - i
            break

    if left_dis == 0:
        for i in range(20, 600, 1):
            if abs(int(np.var(img[50:int(0.6*h),i:i+5])-np.var(img[50:int(0.6*h),15:20]))) > 40:#40
                left_dis = i
                break
    if right_dis == 0:
        for i in range(w-20, w-600, -1):
            if abs(int(np.var(img[50:int(0.6*h),i-5:i])-np.var(img[50:int(0.6*h),w-20:w-15]))) > 40:#80
                right_dis = w - i
                break    
    print("left_dis, right_dis =", left_dis, right_dis)
    
    if abs(left_dis - right_dis) > 0.1*w:
        print('识别错误', filename)
        return 0
    
    if left_dis - right_dis > 0.003*w and left_dis - right_dis > 15:
        # print('向右偏',filename)
        if left_dis - right_dis > 0.015*w:
            print('向右偏,扣2分', filename)
            score_pos_lr = 2
            # copyfile(path + filename, right_path2 + filename)
        else:
            print('向右偏,扣1分', filename)
            score_pos_lr = 1
            # copyfile(path + filename, right_path1 + filename)
    elif right_dis - left_dis > 0.003*w and right_dis - left_dis > 15:
        print('向左偏', filename)
        if right_dis - left_dis > 0.015*w:
            print('向左偏，扣2分', filename)
            score_pos_lr = -2
            # copyfile(path + filename, left_path2 + filename)
        else:
            print('向左偏，扣1分', filename)
            score_pos_lr = -1
            # copyfile(path + filename, left_path1 + filename)
    else:
        print('左右不偏', filename)
        # copyfile(path + filename, standard_path + filename)

    # score_pos_lr: <0表示向左偏，>0表示向右偏
    return score_pos_lr

# 判断上下
def location_ud(img, filename):
    h, w, k = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    down_dis = 0
    up_dis = 0

    score_pos_ud = 0

    for i in range(10,400,1):
        if int(np.var(img[i:i+5,50:int(0.6*h)])-np.var(img[5:10,50:int(0.6*h)])) > 40:#40
            up_dis = i
            break
    for i in range(h-10,h-400,-1):
        if int(np.var(img[i-5:i,50:int(0.6*h)])-np.var(img[h-10:h-5,50:int(0.6*h)])) > 40:#80
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
    print("down_dis, up_dis =", down_dis, up_dis)
    
    if abs(down_dis - up_dis) > 0.1*h:
        print('识别错误', filename)
        return 0
    
    if down_dis - up_dis > 0.003*h:
        # print('向上偏',filename)
        if down_dis - up_dis > 0.015*h:
            print(print('向上偏,扣2分', filename))
            score_pos_ud = 2
            # copyfile(path + filename, up_path2 + filename)
        else:
            print(print('向上偏,扣1分', filename))
            score_pos_ud = 1
            # copyfile(path + filename, up_path1 + filename)
    elif up_dis - down_dis > 0.003*h:
        print('向下偏', filename)
        if up_dis - down_dis > 0.015*h:
            print('向下偏，扣2分', filename)
            score_pos_ud = -2
            # copyfile(path + filename, down_path2 + filename)
        else:
            print('向下偏，扣1分', filename)
            score_pos_ud = -1
            # copyfile(path + filename, down_path1 + filename)
    else:
        print('上下不偏，不扣分', filename)
        # copyfile(path + filename, standard_path + filename)

    # score_pos_ud: <0表示向下偏，>0表示向上偏
    return score_pos_ud
    
def main(file_path):
    file_name = os.path.basename(file_path)
    img = np.fromfile(file_path, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    score_pos_lr = location_lr(img, file_name)
    # score_pos_ud = location_ud(img, file_name)
    # print(score_pos_lr, score_pos_ud)
    # return abs(score_pos_lr), abs(score_pos_ud)
    return abs(score_pos_lr)

    
if __name__ == "__main__":
    time1 = time.time()
    root_path = r"D:/task-7/location/select/cut/up/"
    list_file = os.listdir(root_path)
    for file in list_file:
        if file[-3:] == 'jpg':
            data = np.fromfile(root_path + file, dtype=np.uint8)  
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            location_ud(img, root_path, file)
    time2 = time.time()
    #print('用时：',time2-time1)
