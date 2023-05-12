# -*- coding: UTF-8 -*-

import math
import os
from PIL import Image


def area_count(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y):
    s1 = 0.5 * abs(p0x*p1y+p2x*p0y+p1x*p2y-p2x*p1y-p1x*p0y-p0x*p2y)
    s2 = 0.5 * abs(p1x*p2y+p3x*p1y+p2x*p3y-p3x*p2y-p2x*p1y-p1x*p3y)
    print(s1)
    print(s2)
    return s1+s2


def score_defect_new(img, xyxy_inputs):
    img_h = img.shape[0]
    img_w = img.shape[1]
    imgArea = img_h * img_w

    defect_area = 0
    # conf, poly = label_poly_tuple
    for conf, xyxy in xyxy_inputs:
        x1, y1 = int(xyxy[0]), int(xyxy[1])  # 左上角
        x2, y2 = int(xyxy[2]), int(xyxy[3])  # 右下角
        tmp_area = abs((x2 - x1) * (y2 - y1))
        defect_area += tmp_area

    rate = defect_area / imgArea

    # 缺损对应的阈值[0.01,0.005]
    if rate > 0.01:
        defect_score = 2
        # print("这张邮票分类属于扣2分类")
    elif rate > 0.005:
        defect_score = 1
        # print("这张邮票分类属于扣1分类")
    else:
        defect_score = 0
        # print("这张邮票分类属于扣0分类")

    return defect_score


def score_defect(img_name, save_box_single_dir, img_path):
    img = Image.open(img_path)
    img_h = img.height
    img_w = img.width
    imgArea = img_h * img_w

    path_defect_txt = os.path.join(save_box_single_dir, img_name[:-4]+"__defect.txt")
    if not os.path.exists(path_defect_txt):
        print("something is wrong with path_defect_txt", path_defect_txt)
        return 0

    defect_area = 0
    with open(path_defect_txt, "r", encoding="UTF-8") as fr:
        lines = fr.readlines()
        if len(lines) == 0:
            return 0

        for line in lines:
            lis = line.strip().split(" ")
            x1, y1 = int(lis[1]), int(lis[2])  # 左上角
            x2, y2 = int(lis[3]), int(lis[4])  # 右下角
            tmp_area = abs((x2-x1)*(y2-y1))

            # 应该是这个逻辑
            defect_area += tmp_area

        rate = defect_area/imgArea

        # 缺损对应的阈值[0.01,0.005]
        if rate > 0.01:
            defect_score = 2
            # print("这张邮票分类属于扣2分类")
        elif rate > 0.005:
            defect_score = 1
            # print("这张邮票分类属于扣1分类")
        else:
            defect_score = 0
            # print("这张邮票分类属于扣0分类")

        return defect_score
