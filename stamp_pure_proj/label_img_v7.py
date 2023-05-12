# -*- coding: UTF-8 -*-

import os
import torch
import cv2
from cut_margin import cut_rate_simple as cut_rate
from adjust_perf import extend_rate
import intersection_cal as cal
import adjust_common_v4
# from yolov5_obb_crease.utils_obb.plots import colors
from yolov5_6_1.utils.plots import colors
# 本py文件存放：各小项对图片进行标框的代码

# 正面
# index: 0        1          2         3        4        5           6             7            8          9        10
cls = ["ele", "abrasion", "defect", "stain", "crease", "torn", "ele_damage", "perf_damage", "reflect", "pat_pos", "fade"]
# index:    0       1      2       3      4       5        6         7          8           9        10
cls_ZH = ["要素", "磨损", "缺损", "污渍", "折痕", "撕裂", "要素损坏", "齿孔", "反射效果变暗", "图案位置", "褪色"]
# index:         0          1         2        3        4          5              6
# label_cls = ["abrasion", "defect", "stain", "crease", "torn", "ele_damage", "perf_damage"]


def handle_back_mould_stain(mould_stain_xyxy_list, annotator, isChinese=False):
    for conf, xyxy in mould_stain_xyxy_list:
        if isChinese:
            tmp_label = f'{"背面污渍"} {conf:.2f}'
        else:
            tmp_label = f'{"stain"} {conf:.2f}'
        annotator.box_label(xyxy, label=tmp_label, color=colors(0, True))
    im0 = annotator.result()

    return im0


def handle_ele_overlap(cls_labels_list, img):
    """
    cls_labels_list:    [ele_xyxy_list,
                       abra_real_xyxy_list,
                       defect_xyxy_list,
                       stain_xyxy_list,
                       crease_poly_list,
                       torn_poly_list]
    xyxy_list format: [(conf1, xyxy1), (conf1, xyxy2), ...]
    img: common_adjusted_img
    """
    img_h, img_w, _ = img.shape
    img_area = img_h * img_w

    ele_dmg_xyxy_list = []
    # 把6项小类的rect都放进cls_rect_list
    # cls_rects_list index -> 0:ele, 1:abrasion, 2:defect, 3:stain, 4:crease, 5:torn
    cls_rects_list = [[] for _ in range(6)]
    for i in range(6):
        print(str(cls[i]), cls_rects_list[i])
        for conf, anchor in cls_labels_list[i]:
            # 前4个正常yolov5模型，后两个yolov5_obb模型
            if i <= 3:
                rect = cal.yolo_rectGet_new(anchor)
            else:  # i == 4 or i == 5
                rect = cal.obb_yolo_rectGet_new(anchor)

            cls_rects_list[i].append((conf, rect))
    # print("cls_rects_list", cls_rects_list)

    # index:     0          1         2        3        4          5
    # cls = ["abrasion", "defect", "stain", "crease", "torn", "ele_damage"]

    # 每项与ele的最大重叠区域与那个ele区域的比值
    overlap_max_areas = [float(0) for _ in range(5)]
    # 每项与ele的重叠区域的个数
    overlap_nums = [float(0) for _ in range(5)]
    # 每项与ele的重叠区域之和与图片总面积的比值
    overlap_sum_areas = [float(0) for _ in range(5)]
    for ele_rect_conf, ele_rect in cls_rects_list[0]:
        for i in range(1, 6):
            # 两种特殊情况：1.没检测到某个cls 2.检测到的某个cls框与ele框没有重叠部分
            if len(cls_rects_list[i]) == 0:
                continue
            print("len(cls_rects_list[" + str(i) + "])", len(cls_rects_list[i]))
            for other_rect_conf, other_rect in cls_rects_list[i]:
                tmp_overlap_area, tmp_ele_dmg_rect = cal.intersec_cal(ele_rect, other_rect)
                if not tmp_ele_dmg_rect:
                    continue
                tmp_ele_area = cal.area_cal(ele_rect)
                overlap_max_areas[i - 1] = max(overlap_max_areas[i - 1], tmp_overlap_area / tmp_ele_area)
                overlap_sum_areas[i - 1] += tmp_overlap_area / img_area
                overlap_nums[i - 1] += 1

                # cls_labels_list补全最后一项ele_damage
                anchor_points = cv2.boxPoints(tmp_ele_dmg_rect)
                tmp_ele_dmg_xyxy = [torch.tensor(anchor_points[0][0]), torch.tensor(anchor_points[0][1]),
                                    torch.tensor(anchor_points[2][0]), torch.tensor(anchor_points[2][1])]
                ele_dmg_xyxy_list.append((ele_rect_conf * other_rect_conf, tmp_ele_dmg_xyxy))
    ele_dmg_score_inputs = overlap_nums + overlap_max_areas + overlap_sum_areas

    return ele_dmg_score_inputs, ele_dmg_xyxy_list


def handle_perf_transform(perf_xyxy_list, img):
    """
    perf_xyxy_list:[[(conf, xyxy), ...], [], [] ,[]] 对应四个边
    perf_xyxy_list[0]:上边
    perf_xyxy_list[1]:下边
    perf_xyxy_list[2]:左边
    perf_xyxy_list[3]:右边
    img：common_adjusted_img
    """

    img_h, img_w, _ = img.shape
    perf_real_xyxy_list = []
    for i in range(4):
        if len(perf_xyxy_list[i]) == 0:
            continue
        for conf, xyxy in perf_xyxy_list[i]:
            if i == 0 or i == 2:
                tmp_real_xyxy = xyxy
            elif i == 1:
                tmp_real_xyxy = [xyxy[0] + round(img_h - img_h * cut_rate), xyxy[1],
                                 xyxy[2] + round(img_h - img_h * cut_rate), xyxy[3]]
            else:   # i == 3
                tmp_real_xyxy = [xyxy[0], xyxy[1] + round(img_w - img_w * cut_rate),
                                 xyxy[2], xyxy[3] + round(img_w - img_w * cut_rate)]
            perf_real_xyxy_list.append((conf, tmp_real_xyxy))

    return perf_real_xyxy_list


def handle_perf_trans_with_extend(perf_xyxy_list, img_com, img_perf):
    """
    perf_xyxy_list:[[(conf, xyxy), ...], [], [] ,[]] 对应四个边
    perf_xyxy_list[0]:上边
    perf_xyxy_list[1]:下边
    perf_xyxy_list[2]:左边
    perf_xyxy_list[3]:右边
    img：common_adjusted_img
    """

    com_h, com_w = img_com.shape[:2]
    com_min = min(com_h, com_w)
    perf_h, perf_w = img_perf.shape[:2]
    extend_len = round(com_min * extend_rate)
    cut_len = round(com_min * cut_rate)

    perf_real_xyxy_list = []
    for i in range(4):
        for conf, xyxy in perf_xyxy_list[i]:
            if i == 0 or i == 2:
                tmp_real_xyxy = [xyxy[0] - extend_len, xyxy[1] - extend_len,
                                 xyxy[2] - extend_len, xyxy[3] - extend_len]
            elif i == 1:
                tmp_real_xyxy = [xyxy[0] + (perf_h - cut_len - extend_len), xyxy[1] - extend_len,
                                 xyxy[2] + (perf_h - cut_len - extend_len), xyxy[3] - extend_len]
            else:   # i == 3
                tmp_real_xyxy = [xyxy[0] - extend_len, xyxy[1] + (perf_w - cut_len - extend_len),
                                 xyxy[2] - extend_len, xyxy[3] + (perf_w - cut_len - extend_len)]
            perf_real_xyxy_list.append((conf, tmp_real_xyxy))

    return perf_real_xyxy_list


def handle_crease_remove_margin(crease_poly_list, img_com):
    thres_rate = 0.05
    com_h, com_w = img_com.shape[:2]
    com_min = min(com_h, com_w)

    com_thres = com_min * thres_rate

    crease_real_poly_list = []
    for conf, poly in crease_poly_list:
        x1, y1, x2, y2, x3, y3, x4, y4 = poly
        x_min = min(float(x1), float(x2), float(x3), float(x4))
        x_max = max(float(x1), float(x2), float(x3), float(x4))
        y_min = min(float(y1), float(y2), float(y3), float(y4))
        y_max = max(float(y1), float(y2), float(y3), float(y4))
        # 左边界
        if x_min < 50 and x_max < com_thres and (y_max - y_min ) > 3 * (x_max - x_min):
            continue
        # 右边界
        if (com_w - x_max) < 50 and (com_w - x_min) < com_thres and  (y_max - y_min) > 3 * (x_max - x_min):
            continue
        # 上边界
        if y_min < 50 and y_max < com_thres and (x_max - x_min) > 3 * (y_max - y_min):
            continue
        # 下边界
        if (com_h - y_max) < 50 and (com_h - y_min) < com_thres and (x_max - x_min) > 3 * (y_max - y_min):
            continue
        crease_real_poly_list.append((conf, poly))

    return crease_real_poly_list

def handle_abra_cvt_pos(abra_xyxy_list, mat_com):
    abra_real_xyxy_list = []
    for abra_conf, abra_xyxy in abra_xyxy_list:
        point1 = (int(abra_xyxy[0]), int(abra_xyxy[1]))
        point2 = (int(abra_xyxy[2]), int(abra_xyxy[3]))
        trans_point1 = adjust_common_v4.cvt_pos(point1, mat_com)
        trans_point2 = adjust_common_v4.cvt_pos(point2, mat_com)
        abra_trans_xyxy = [torch.tensor(trans_point1[0]), torch.tensor(trans_point1[1]),
                           torch.tensor(trans_point2[0]), torch.tensor(trans_point2[1])]
        abra_real_xyxy_list.append((abra_conf, abra_trans_xyxy))

    return abra_real_xyxy_list


def label_all_cls(cls_labels_list, annotator, isChinese=False):
    """
    cls_labels_list:[   ele_xyxy_list,          0
                        abra_real_xyxy_list,    1
                        defect_xyxy_list,       2
                        stain_xyxy_list,        3
                        crease_real_poly_list,  4
                        torn_poly_list,         5
                        ele_dmg_xyxy_list,      6
                        perf_real_xyxy_list]    7
    annotator: 标框
    """
    for i in range(1, len(cls_labels_list)):
        for conf, xyxy in cls_labels_list[i]:
            if isChinese:
                tmp_label = f'{cls_ZH[i]} {conf:.2f}'
            else:
                tmp_label = f'{cls[i]} {conf:.2f}'
            print("tmp_label:", tmp_label)

            if i < 4 or i > 5:
                annotator.box_label(xyxy, label=tmp_label, color=colors((i-1) * 2, True))
            else:  # i == 4 or i == 5
                poly = xyxy
                annotator.poly_label(poly, label=tmp_label, color=colors((i-1) * 2, True))
    im0 = annotator.result()

    return im0

