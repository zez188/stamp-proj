# -*- coding: UTF-8 -*-
import numpy as np
import cv2
# import time
# start=time.time()

def yolo_rectGet_new(xyxy: list) -> np.ndarray:
    cnt = np.array(np.float32([[float(xyxy[0]), float(xyxy[1])],
                               [float(xyxy[0]), float(xyxy[3])],
                               [float(xyxy[2]), float(xyxy[3])],
                               [float(xyxy[2]), float(xyxy[1])]]))  # 必须是array数组的形式
    rect = cv2.minAreaRect(cnt)                                     # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
    return rect

def obb_yolo_rectGet_new(poly: list) -> np.ndarray:
    cnt = np.array(np.float32([[float(poly[0]), float(poly[1])],
                               [float(poly[2]), float(poly[3])],
                               [float(poly[4]), float(poly[5])],
                               [float(poly[6]), float(poly[7])]]))  # 必须是array数组的形式
    rect = cv2.minAreaRect(cnt)                                     # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
    return rect


def yolo_rectGet(xyxy: list) -> np.ndarray: # x1, y1, x2, y2

    # 改了，xyxy[0]是conf
    cnt = np.array(np.float32([[xyxy[1], xyxy[2]],
                               [xyxy[1], xyxy[4]],
                               [xyxy[3], xyxy[4]],
                               [xyxy[3], xyxy[2]]]))  # 必须是array数组的形式

    rect = cv2.minAreaRect(cnt)
    # print(rect)
    return rect

def rotate_yolo_rectGet(poly: list) -> np.ndarray:
    # cnt = np.array(np.float32([[poly[0], poly[1]], [poly[2], poly[3]], [poly[4], poly[5]],
    #                            [poly[6], poly[7]]]))  # 必须是array数组的形式
    # 改了，xyxy[0]是conf
    cnt = np.array(np.float32([[poly[1], poly[2]],
                               [poly[3], poly[4]],
                               [poly[5], poly[6]],
                               [poly[7], poly[8]]]))  # 必须是array数组的形式
    rect = cv2.minAreaRect(cnt)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
    return rect

def intersec_cal(rect1, rect2):
    # rect1 = ((0,0),(1,1),45)
    # rect2 = ((1.5,0),(4,3),0)

    r1 = cv2.rotatedRectangleIntersection(rect2, rect1)
    # print("r1", r1)
    # print("type(r1[1])", type(r1[1]))
    if type(r1[1]) != np.ndarray:
        area = 0
        # print("intersection area: ", area)
        return area, None

    area = cv2.contourArea(r1[1])

    ele_anchor = cv2.minAreaRect(r1[1])
    # print("intersection area: ", area)
    return area, ele_anchor

def area_cal(rect):
    w, h = rect[1]
    area = w*h
    # print("rect area: ", area)
    return area

# print(time.time()-start)

if __name__ == "__main__":
    rect1 = ((0,0),(1,1),45)
    rect2 = ((1.5,0),(4,3),0)
    intersec_cal(rect1, rect2)
