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


def score_torn_new(img, label_poly_inputs):
    img_h = img.shape[0]
    img_w = img.shape[1]
    imgArea = img_h * img_w

    torn_area = 0
    # conf, poly = label_poly_tuple
    for conf, poly in label_poly_inputs:
        p0x, p0y = int(float(poly[0])), int(float(poly[1]))  # 左上角
        p1x, p1y = int(float(poly[2])), int(float(poly[3]))  # 右上角
        p2x, p2y = int(float(poly[4])), int(float(poly[5]))  # 左下角
        p3x, p3y = int(float(poly[6])), int(float(poly[7]))  # 右下角

        tmp_area = area_count(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y)
        a = math.sqrt((p0y - p1y) * (p0y - p1y) + (p0x - p1x) * (p0x - p1x))
        b = math.sqrt((p0y - p2y) * (p0y - p2y) + (p0x - p2x) * (p0x - p2x))

        longside = max(a, b)
        shortside = min(a, b)

        if longside> 0 and shortside < longside:
            torn_area += tmp_area

    rate = torn_area / imgArea

    # 撕裂对应的阈值[0.01,0.00032]
    if rate > 0.01:
        score = 2
        # print("这张邮票分类属于扣2分类")
    elif rate > 0.00032:
        score = 1
        # print("这张邮票分类属于扣1分类")
    else:
        score = 0
        # print("这张邮票分类属于扣0分类")

    return score


def score_torn(img_name, save_box_single_dir, img_path):

    img = Image.open(img_path)
    img_h = img.height
    img_w = img.width
    imgArea = img_h * img_w

    path_torn_txt = os.path.join(save_box_single_dir, img_name[:-4]+"__torn.txt")
    if not os.path.exists(path_torn_txt):
        print("something is wrong with path_torn_txt", path_torn_txt)
        return 0

    torn_area = 0
    with open(path_torn_txt, "r", encoding="UTF-8") as fr:
        lines = fr.readlines()
        if len(lines) == 0:
            return 0

        for line in lines:
            lis = line.strip().split(" ")

            p0x, p0y = int(float(lis[1])), int(float(lis[2]))  # 左上角
            p1x, p1y = int(float(lis[3])), int(float(lis[4]))  # 右上角
            p2x, p2y = int(float(lis[5])), int(float(lis[6]))  # 左下角
            p3x, p3y = int(float(lis[7])), int(float(lis[8]))  # 右下角

            torn_area += area_count(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y)

        rate = torn_area / imgArea

        # 撕裂对应的阈值[0.01,0.00032]
        if rate > 0.01:
            score = 2
            # print("这张邮票分类属于扣2分类")
        elif rate > 0.00032:
            score = 1
            # print("这张邮票分类属于扣1分类")
        else:
            score = 0
            # print("这张邮票分类属于扣0分类")

        return score