# -*- coding: UTF-8 -*-

import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import sys

# from yolov5_6_1.utils.plots import Annotator, colors
from yolov5_obb_crease.utils_obb.plots import Annotator, colors

from yolov5_obb_crease import detect_crease, detect_torn
from yolov5_6_1 import detect_ele_1, detect_abrasion, detect_defect, detect_stain, detect_perforation_4, detect_refl

from yolov5_6_1 import score_abrasion, score_defect, score_stain, score_ele_damage, score_perforation, score_refl_darken
from yolov5_obb_crease import score_crease_mark, score_torn

from fade_yellowing import svm_yellowing_new, svm_tuise
from pattern_pos import score_pat_pos

import label_img_v7

# import adjust_common_v4

save_anchored_img_dir = "save_anchored_img_dir"

# 正面
# index  0        1          2         3        4        5           6             7            8          9        10
cls = ["ele", "abrasion", "defect", "stain", "crease", "torn", "ele_damage", "perf_damage", "reflect", "pat_pos", "fade"]
# index          0          1         2        3        4           5             6            7          8        9
cls_front = ["abrasion", "defect", "stain", "crease", "torn", "ele_damage", "perf_damage", "reflect", "pat_pos", "fade"]
cls_front_ZH = ["磨损", "缺损", "污渍", "折痕", "撕裂", "票面要素", "齿孔", "反射效果变暗", "图案位置", "褪色"]
# 背面
# index          0              1
cls_back = ['back_yellow', 'back_stain']
cls_back_ZH = ["背面泛黄", "背面污渍"]
# front index:          0          1         2        3        4          5              6            7          8        9
# score_cls_list = ["abrasion", "defect", "stain", "crease", "torn", "ele_damage", "perf_damage", "reflect", "pat_pos", "fade"]


def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)  # -1 表示用原通道方式读取图片
    return cv_img


def cv_imwrite(pic_path, pic):
    # x1部分填写相应的图片类型，例如.jpg .png等等，x2部分填写图片，x3部分填写路径及名字。
    cv2.imencode('.jpg', pic)[1].tofile(pic_path)


# 正面检测、标框、打分
def detectFront(file_ori_path, file_com_path, file_pos_path, file_per_path, file_perf_dir, file_cls_name, mat_com, use_ZH):
    """
    file_ori_path: ori_pic's abs_path
    file_com_path: common_adjusted_pic's abs_path
    file_pos_path: pat_pos_adjusted_pic's abs_path
    file_per_path: perf_adjusted_pic's abs_path
    file_perf_dir: perf_adjusted_pic, cut its margins into 4 margin_pics, then save in this dir. abs_path format.
    file_cls_name: pic's standard name
    mat_com: transformation matrix, from ori_pic to common_adjusted_pic
    use_ZH: if use chinese to label pic or not
    """
    os.makedirs(save_anchored_img_dir, exist_ok=True)
    # print("file_ori_path", file_ori_path)
    # print("file_com_path", file_com_path)
    # print("file_pos_path", file_pos_path)
    # print("file_per_path", file_per_path)
    # print("file_perf_dir", file_perf_dir)

    # img_com = cv_imread(file_com_path)
    # img_com = cv2.imdecode(np.fromfile(file_com_path, dtype=np.uint8), -1)
    img_com = cv_imread(file_com_path)
    img_com_copy = img_com.copy()

    img_perf = cv_imread(file_per_path)

    img_name = os.path.basename(file_com_path)

    ###########################################################################################

    # yolo识别部分

    os.chdir("./yolov5_6_1")

    # 票面要素执行 获取标记信息
    ele_xyxy_list = detect_ele_1.main(file_com_path)
    # 磨损执行 获取标记信息
    abra_score_inputs, abra_xyxy_list = detect_abrasion.main(file_ori_path)
    # 缺损执行 获取标记信息
    defect_xyxy_list = detect_defect.main(file_com_path)
    # 污渍、黄斑执行 获取标记信息
    stain_score_inputs, stain_xyxy_list, mould_stain_xyxy_list = detect_stain.main(file_com_path)
    # 齿孔执行 获取标记信息
    perf_xyxy_list = detect_perforation_4.main(file_perf_dir)

    # 反射效果变暗执行 获取标记信息
    refl_c0_xyxy_list, refl_c1_xyxy_list = detect_refl.main(file_com_path)
    refl_xyxy_list = refl_c0_xyxy_list + refl_c1_xyxy_list
    # print("refl_xyxy_list", refl_xyxy_list)

    os.chdir("../yolov5_obb_crease")

    # 折痕执行 获取标记信息
    # crease_poly_list, poly_cla_list = [], []
    crease_poly_list, poly_cla_list = detect_crease.main(file_com_path)

    # 撕裂执行 获取标记信息
    # torn_poly_list = []
    torn_poly_list = detect_torn.main(file_com_path)

    os.chdir("..")

    ##################################################################################################

    # 标框部分

    # line_thickness = 5
    # anno = Annotator(img_com_copy, line_width=line_thickness, example="汉字")
    if use_ZH:
        anno = Annotator(img_com_copy, line_width=5, font_size=50, example="汉字")
    else:
        anno = Annotator(img_com_copy, line_width=5, font_size=5)

    # 磨损的在原图上的标框位置信息转换为在矫正图上的标框信息
    abra_real_xyxy_list = label_img_v7.handle_abra_cvt_pos(abra_xyxy_list, mat_com)
    # 折痕的去掉一些在边界有误检嫌疑的框
    crease_real_poly_list = label_img_v7.handle_crease_remove_margin(crease_poly_list, img_com)

    cls_labels_list = [ele_xyxy_list,
                       abra_real_xyxy_list,
                       defect_xyxy_list,
                       stain_xyxy_list,
                       crease_real_poly_list,
                       torn_poly_list]
    # print("cls_labels_list before", cls_labels_list)

    # handle_ele_overlap() 计算：ele与其他5小类的重叠信息（ele_damage的标框位置信息）+ ele_damage打分所需inputs
    ele_dmg_score_inputs, ele_dmg_xyxy_list = label_img_v7.handle_ele_overlap(cls_labels_list, img_com)
    cls_labels_list.append(ele_dmg_xyxy_list)

    # handle_perf_transform() 计算：齿孔四条边的标框位置转信息转换为矫正图上的标框位置信息
    perf_real_xyxy_list = label_img_v7.handle_perf_trans_with_extend(perf_xyxy_list, img_com, img_perf)
    cls_labels_list.append(perf_real_xyxy_list)
    cls_labels_list.append(refl_xyxy_list)
    cls_labels_list[3] = mould_stain_xyxy_list

    # label_all_cls() 执行所有框在矫正图上的标注
    labelled_img = label_img_v7.label_all_cls(cls_labels_list, anno, isChinese=True)

    anchored_img_front_name = img_name[:-4] + "_frontAnchored.jpg"
    write_front_path = os.path.join(save_anchored_img_dir, anchored_img_front_name)
    write_front_path = os.path.abspath(write_front_path)
    print("write_front_path", write_front_path)
    cv_imwrite(write_front_path, labelled_img)

    #####################################################################################################

    # 打分部分

    # 磨损打分
    abra_score = score_abrasion.grade_abrasion(abra_score_inputs)
    # 缺损打分
    defect_score = score_defect.score_defect_new(img_com, defect_xyxy_list)
    # 污渍打分
    stain_score = score_stain.grade_stain_mould(stain_score_inputs)
    # 折痕打分
    crease_score = score_crease_mark.score_crease_new(img_com, poly_cla_list)
    # 撕裂打分
    torn_score = score_torn.score_torn_new(img_com, torn_poly_list)
    # 票面要素打分
    ele_dmg_score = score_ele_damage.score_ele_dmg_new(ele_dmg_score_inputs)
    # 齿孔打分
    perf_score = score_perforation.score_perf_new(img_com, perf_xyxy_list)
    # 反射效果变暗打分
    refl_score = score_refl_darken.score_refl(refl_c0_xyxy_list, refl_c1_xyxy_list)

    # 图案位置打分
    pos_lr_score = score_pat_pos.main(file_pos_path)
    if file_cls_name and "T46.1-1" in file_cls_name:
        if pos_lr_score >= 0:
            pos_lr_score = min(pos_lr_score, 1)
        else:
            pos_lr_score = max(pos_lr_score, -1)
    # 褪色打分
    fade_score = svm_tuise.tuise(file_com_path)

    #####################################################################################################

    # 录入分数部分
    
    score_front_list = [cls_front[i] + ":" + str(0) for i in range(len(cls_front))]
    # score_front_ZH_list = [cls_front[i] + ":" + str(0) for i in range(len(cls_front))]

    score_front_nums = [abra_score,          # 0
                        defect_score,        # 1
                        stain_score,         # 2
                        crease_score,        # 3
                        torn_score,          # 4
                        ele_dmg_score,       # 5
                        perf_score,          # 6
                        refl_score,          # 7
                        pos_lr_score,        # 8
                        fade_score]          # 9
    # use_ZH = True
    if use_ZH:
        for i in range(len(cls_front_ZH)):
            if i == 8:
                if score_front_nums[i] > 0:
                    score_front_list[i] = cls_front_ZH[i] + "右偏" + ":" + str(score_front_nums[i])
                elif score_front_nums[i] < 0:
                    score_front_list[i] = cls_front_ZH[i] + "左偏" + ":" + str(score_front_nums[i])
                else:  # pos_lr_score = 0
                    score_front_list[i] = cls_front_ZH[i] + ":" + str(0)
            else:
                score_front_list[i] = cls_front_ZH[i] + ":" + str(score_front_nums[i])
    else:
        for i in range(len(cls_front)):
            if i == 8:
                if score_front_nums[i] > 0:
                    score_front_list[i] = cls_front[i] + "右偏" + ":" + str(score_front_nums[i])
                elif score_front_nums[i] < 0:
                    score_front_list[i] = cls_front[i] + "左偏" + ":" + str(score_front_nums[i])
                else:  # pos_lr_score = 0
                    score_front_list[i] = cls_front[i] + ":" + str(0)
            else:
                score_front_list[i] = cls_front[i] + ":" + str(score_front_nums[i])

    return write_front_path, score_front_list


# 背面检测、标框、打分
def detectBack(file_com_path, file_cls_name, use_ZH):
    """
    file_com_path: common_adjusted back_pic's abs_path
    file_cls_name: pic's standard name
    use_ZH: if use chinese to label pic or not
    """
    # start = time.time()

    # index          0              1
    # cls_back = ['back_yellow', 'back_stain']
    score_back_list = [cls_back[i] + ':' + str(0) for i in range(len(cls_back))]

    img_name = os.path.basename(file_com_path)

    img_com = cv_imread(file_com_path)
    img_com_copy = img_com.copy()

    # line_thickness = 5
    if use_ZH:
        anno_back = Annotator(img_com_copy, line_width=5, font_size=50, example="汉字")
    else:
        anno_back = Annotator(img_com_copy, line_width=5, font_size=5)
    # img_com_copy, line_width = line_thickness, font_size = 5
    ##################################################################################

    # 标框、打分部分

    os.chdir("./yolov5_6_1")

    # 污渍、黄斑执行 获取框位置信息
    stain_score_inputs, stain_xyxy_list, mould_stain_xyxy_list = detect_stain.main(file_com_path, "back")

    os.chdir("..")

    # 污渍执行 打分
    back_stain_score = score_stain.grade_stain_mould(stain_score_inputs)
    if file_cls_name and "T46.1-1" in file_cls_name:
        back_stain_score = 0
    # 泛黄执行 打分
    back_yellow_score = svm_yellowing_new.score_yellowing(file_com_path)

    # 污渍、黄斑执行 标框
    anchored_back_img = label_img_v7.handle_back_mould_stain(stain_xyxy_list, anno_back, use_ZH)

    # 保存标注后的矫正图
    anchored_img_back_name = img_name[:-4] + "_backAnchored.jpg"
    write_back_path = os.path.join(save_anchored_img_dir, anchored_img_back_name)
    # print("write_back_path", write_back_path)
    cv_imwrite(write_back_path, anchored_back_img)

    # print("detectBack() totally spend time:", time.time() - start)

    #####################################################################################################

    # 整合分数score_front_list

    # use_ZH = True
    if use_ZH:
        # 背泛黄分数输入
        score_back_list[0] = cls_back_ZH[0] + ':' + str(back_yellow_score)
        # 背污渍分数输入
        score_back_list[1] = cls_back_ZH[1] + ':' + str(back_stain_score)
    else:
        # 背泛黄分数输入
        score_back_list[0] = cls_back[0] + ':' + str(back_yellow_score)
        # 背污渍分数输入
        score_back_list[1] = cls_back[1] + ':' + str(back_stain_score)

    return write_back_path, score_back_list


if __name__ == "__main__":
    file_com_path = r"D:\PycharmProjects\opencv_handle_364\yolov5_ele\mydata\detect_imgs\1994-18（6-5）3.jpg"
    detectBack(file_com_path)

