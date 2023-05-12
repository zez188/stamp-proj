# -*- coding: UTF-8 -*-

import cv2
import math
import numpy as np
import os

# 超参数，控制齿孔边缘切割部分大小
cut_rate_adjusted = 0.08
cut_rate_ori = 0.15
cut_rate_simple = 0.1


def cut_margin_simple(img_path):

    img_name = os.path.basename(img_path)
    # save_box_single_dir = os.path.join(save_box_info_dir, img_name[:-4])
    save_all_margins_dir = "save_margins_dir"
    save_margins_single_dir = os.path.join(save_all_margins_dir, img_name[:-4])
    print("save_margins_single_dir", save_margins_single_dir)
    os.makedirs(save_margins_single_dir, exist_ok=True)

    ori_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)

    # n表示高，m表示宽
    # print(ori_img.shape[0], ori_img.shape[1])
    h = ori_img.shape[0]
    w = ori_img.shape[1]

    cv2.imencode('.jpg', ori_img[:int(cut_rate_simple*h), :])[1].tofile('{}/{}'.format(save_margins_single_dir, img_name[:-4] + "_1.jpg"))
    cv2.imencode('.jpg', ori_img[int((1-cut_rate_simple)*h):, :])[1].tofile('{}/{}'.format(save_margins_single_dir, img_name[:-4] + "_2.jpg"))
    cv2.imencode('.jpg', ori_img[:, :int(cut_rate_simple*w)])[1].tofile('{}/{}'.format(save_margins_single_dir, img_name[:-4] + "_3.jpg"))
    cv2.imencode('.jpg', ori_img[:, int((1-cut_rate_simple)*w):])[1].tofile('{}/{}'.format(save_margins_single_dir, img_name[:-4] + "_4.jpg"))

    return os.path.abspath(save_margins_single_dir)


def cut_perf_adj_margin(com_img_path, perf_img_path):

    img_name = os.path.basename(perf_img_path)
    save_all_margins_dir = "save_margins_dir"
    save_margins_single_dir = os.path.join(save_all_margins_dir, img_name[:-4])
    os.makedirs(save_margins_single_dir, exist_ok=True)

    com_img = cv2.imdecode(np.fromfile(com_img_path, dtype=np.uint8), -1)
    perf_img = cv2.imdecode(np.fromfile(perf_img_path, dtype=np.uint8), -1)
    # n表示高，m表示宽
    # print(adjusted_img.shape[0], adjusted_img.shape[1])
    h_com, w_com = com_img.shape[:2]
    com_min = min(h_com, w_com)
    cut_len = round(com_min * cut_rate_simple)
    
    # 拓边后的h_perf，w_perf要比h_com, w_com多extend_len = com_min * extend_rate
    h_perf, w_perf = perf_img.shape[:2]

    cv2.imencode('.jpg', perf_img[:cut_len, :])[1].tofile('{}/{}'.format(save_margins_single_dir, img_name[:-4] + "_1.jpg"))
    cv2.imencode('.jpg', perf_img[h_perf-cut_len:, :])[1].tofile('{}/{}'.format(save_margins_single_dir, img_name[:-4] + "_2.jpg"))
    cv2.imencode('.jpg', perf_img[:, :cut_len])[1].tofile('{}/{}'.format(save_margins_single_dir, img_name[:-4] + "_3.jpg"))
    cv2.imencode('.jpg', perf_img[:, w_perf-cut_len:])[1].tofile('{}/{}'.format(save_margins_single_dir, img_name[:-4] + "_4.jpg"))

    return os.path.abspath(save_margins_single_dir)


# [已弃用]
def cut_adjuested_margin(adjusted_img_path):
    save_box_info_dir = r"save_box_info_dir"
    if not os.path.exists(save_box_info_dir):
        os.mkdir(save_box_info_dir)
    img_name = os.path.basename(adjusted_img_path)
    save_all_margins_dir = "save_margins_dir"
    save_margins_single_dir = os.path.join(save_all_margins_dir, img_name[:-4])
    os.makedirs(save_margins_single_dir, exist_ok=True)

    adjusted_img = cv2.imdecode(np.fromfile(adjusted_img_path, dtype=np.uint8), -1)

    # n表示高，m表示宽
    print(adjusted_img.shape[0], adjusted_img.shape[1])
    h = adjusted_img.shape[0]
    w = adjusted_img.shape[1]

    com_min = min(h, w)

    cv2.imencode('.jpg', adjusted_img[:int(cut_rate_adjusted * com_min), :])[1].tofile('{}/{}'.format(save_margins_single_dir, img_name[:-4] + "_1.jpg"))
    cv2.imencode('.jpg', adjusted_img[h-int(cut_rate_adjusted * com_min):, :])[1].tofile('{}/{}'.format(save_margins_single_dir, img_name[:-4] + "_2.jpg"))
    cv2.imencode('.jpg', adjusted_img[:, :int(cut_rate_adjusted * com_min)])[1].tofile('{}/{}'.format(save_margins_single_dir, img_name[:-4] + "_3.jpg"))
    cv2.imencode('.jpg', adjusted_img[:, w-int(cut_rate_adjusted * com_min):])[1].tofile('{}/{}'.format(save_margins_single_dir, img_name[:-4] + "_4.jpg"))


if __name__ == "__main__":
    path = r"2013-25(4-3).jpg"
    # cut_adjuested_margin(path)
    cut_margin_simple(path)