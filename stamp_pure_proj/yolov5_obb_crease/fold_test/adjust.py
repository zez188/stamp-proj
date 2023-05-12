import cv2
import math
import numpy as np
import os
import matplotlib.pyplot as plt
# from PIL import Image
# import imageio
# from fold_hub import deal_holds

#前处理多了一个处理齿孔的函数，deal_holds


def deal_holds(ori_img):
    h,w,k=ori_img.shape
    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    holds_edg=[]
    up,down,left,right=int(0.3*h),int(0.7*h),int(0.3*w),int(0.7*w)
    for i in range(0,up,5):
        if sum(img[i,left:right]<70)<30:
            holds_edg.append(i+10)
            break

    for i in range(down,h,5)[::-1]:
        if sum(img[i,left:right]<70)<30:
            holds_edg.append(i-10)
            break

    for i in range(0,left,5):
        if sum(img[up:down,i]<70)<30:
            holds_edg.append(i+10)
            break
    for i in range(right,w,5)[::-1]:
        if sum(img[up:down,i]<70)<30:
            holds_edg.append(i-10)
            break
    #holds_edg 上边界，下边界，左边界，右边界
    fine_img=ori_img[holds_edg[0]:holds_edg[1],holds_edg[2]:holds_edg[3]]
    # plt.imshow(fine_img,cmap='gray')
    # plt.show()
    return fine_img

def Img_Outline(input_dir):
    # original_img = cv2.imread(input_dir)
    # original_img=cv2.resize(original_img_01,(600,600),interpolation=cv2.INTER_CUBIC)
    original_img=cv2.imdecode(np.fromfile(input_dir,dtype=np.uint8),-1)
    # original_img=cv2.resize(original_img,(600,800),interpolation=cv2.INTER_CUBIC).astype(np.uint8)


    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_img, (9, 9), 0)                     # 高斯模糊去噪
    #_, RedThresh = cv2.threshold(blurred, 165, 255, cv2.THRESH_BINARY)  # 设定阈值120

    # 改
    _, RedThresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))          # 定义矩形结构元素
    closed = cv2.morphologyEx(RedThresh, cv2.MORPH_CLOSE, kernel)       # 闭运算（链接块）
    opened = cv2.morphologyEx(RedThresh, cv2.MORPH_OPEN, kernel,10)           # 开运算（去噪点）


    return original_img, gray_img, RedThresh, closed, opened


def findContours_img(original_img, opened):
    contours, hierarchy = cv2.findContours(opened, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 改
    h, w = original_img.shape[:2]
    total_area = h * w

    # c = sorted(contours, key=cv2.contourArea, reverse=True)[0]   # 计算最大轮廓的旋转包围盒
    c_list = sorted(contours, key=cv2.contourArea, reverse=True)  # 计算最大轮廓的旋转包围盒
    print("c_list[0].shape", c_list[0].shape)
    print("type", type(c_list[0]))
    # c_real_list = []
    # c_arr = np.empty(shape=(0, 1, 2), dtype=np.float32)
    c_arr = np.empty(shape=(0, 1, 2), dtype=np.float32)
    # c_arr = c_arr.astype(np.float32)
    print("c_arr.shape", c_arr.shape)

    c_arr_count = 0
    for c in c_list:
        # print("c.shape", c.shape)
        tmp_area = cv2.contourArea(c)

        # 设阈值
        if tmp_area/total_area > 1/20:
            c_arr = np.concatenate((c_arr, c), axis=0)
            # c_real_list.extend(c)
            c_arr_count += 1

    c_arr = c_arr.astype(np.float32)
    print("c_arr.dtype", c_arr.dtype)
    # print("len(c_real_list)", len(c_real_list))
    print("c_arr_count", c_arr_count)
    print("len(c_list) = ", len(c_list))
    # print("c:", c)
    rect = cv2.minAreaRect(c_arr)                                    # 获取包围盒（中心点，宽高，旋转角度）

    # if rect[2] > float(45):
    #     rect_other = (rect[0], rect[1], 90-rect[2])
    #     box = np.int0(cv2.boxPoints(rect))  # box
    #     print("rect_other:", rect_other)
    #     print("box[0]:", box[0])   # 左上
    #     print("box[1]:", box[1])   # 左下
    #     print("box[2]:", box[2])   # 右下
    #     print("box[3]:", box[3])   # 左上
    #     draw_img = cv2.drawContours(original_img.copy(), [box], -1, (0, 0, 255), 1)
    #     return box, draw_img, rect_other
    # else:
    box = np.int0(cv2.boxPoints(rect))                           # box
    draw_img = cv2.drawContours(original_img.copy(), [box], -1, (0, 0, 255), 1)
    print("rect:", rect)
    # print("box[0]:", box[0])   # 左下
    # print("box[1]:", box[1])   # 左上
    # print("box[2]:", box[2])   # 右上
    # print("box[3]:", box[3])   # 右下
    return box, draw_img, rect

def Perspective_transform(box,original_img,rect):

    # if rect[2]>-45:
    if rect[2] > 45:
        pts1 = np.float32([box[0], box[1], box[2], box[3]])
        orignal_W = math.ceil(np.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2))
        orignal_H = math.ceil(np.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2))

    else:
        pts1 = np.float32([box[1], box[2], box[3], box[0]])
        orignal_W = math.ceil(np.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2))
        orignal_H = math.ceil(np.sqrt((box[0][1] - box[1][1]) ** 2 + (box[0][0] - box[1][0]) ** 2))

    # 原图中的四个顶点,与变换矩阵
    # pts2 = np.float32([[int(orignal_W+1), int(orignal_H+1)], [0, int(orignal_H+1)], [0, 0], [int(orignal_W+1), 0]])
    pts2 = np.float32([[0, 0], [int(orignal_W+1), 0], [int(orignal_W+1), int(orignal_H+1)], [0, int(orignal_H+1)]])

    # print("pts1:", pts1)
    # print("pts2:", pts2)
    # print(orignal_H, orignal_W, "\n")

    # 生成透视变换矩阵；进行透视变换
    M = cv2.getPerspectiveTransform(pts1, pts2)
    result_img = cv2.warpPerspective(original_img, M, (int(orignal_W+1), int(orignal_H+3)))

    return result_img

#主函数，更改路径就行
if __name__=="__main__":
    # input_dir = "adjust_test/T135（3-2）10.jpg"
    root_path= r'C:\Users\Administrator\Desktop\测评样5.13'
    out_path=r'C:\Users\Administrator\Desktop\company_test_sample'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    list_file=os.listdir(root_path)
    list_file.sort()
    list_file_give=list_file
    if not isinstance(list_file_give,list):
        list_file_give=[list_file_give]
    # for file_name in list_file_give:
    #     img_path=os.path.join(root_path,file_name)
    #     original_img, gray_img, RedThresh, closed, opened = Img_Outline(img_path)
    #     box, draw_img,recct = findContours_img(original_img,opened)
    #     result_img = Perspective_transform(box,original_img,recct)
    #     # plt.imshow(opened)
    #     # plt.show()
    #     # cv2.imwrite(os.path.join(out_path,file_name), result_img.astype(np.uint8))
    #     cv2.imencode('.jpg', result_img)[1].tofile('{}/{}'.format(out_path,file_name))

    for file_name in list_file_give:
        img_path=os.path.join(root_path,file_name)
        # original_img=cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1)
        original_img, gray_img, RedThresh, closed, opened = Img_Outline(img_path)
        box, draw_img,recct = findContours_img(original_img,opened)
        result_img = Perspective_transform(box,original_img,recct)
        result_img=deal_holds(result_img)
        # plt.imshow(opened)
        # plt.show()
        # cv2.imwrite(os.path.join(out_path,file_name), result_img.astype(np.uint8))
        cv2.imencode('.jpg', result_img)[1].tofile('{}/{}'.format(out_path,file_name))


