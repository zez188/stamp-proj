import os
import cv2
import numpy as np


def score_crease_new(img, label_poly_cla_inputs):
    # row:高度 col:宽度
    row, col, _ = img.shape
    mark = 0
    flag = 0
    crease_length = 0
    crease_conf = 0

    # 阈值选取
    w_threshold, h_threshold = row * 0.2, col * 0.2

    for conf, poly, cla in label_poly_cla_inputs:
        data_line = [conf]
        for i in range(8):
            data_line.append(float(poly[i]))
        data_line.append(cla)

        if data_line[-1] == 1:
            if data_line[0] > 0.3:
                mark = 1
            continue
        if data_line[0] < 0.05:
            continue

        # 求矩形最长边
        side_1 = pow(pow(data_line[3] - data_line[1], 2) + pow(data_line[4] - data_line[2], 2), 0.5)
        side_2 = pow(pow(data_line[5] - data_line[1], 2) + pow(data_line[6] - data_line[2], 2), 0.5)
        max_side = max(side_2, side_1)
        crease_length += max_side

        side_check_w, side_check_h = [], []
        for i in range(1, 8, 2):
            if (w_threshold < data_line[i] < col - w_threshold) or (h_threshold < data_line[i + 1] < row - h_threshold):
                flag = 1
                # crease_conf += data_line[-1]
                crease_conf += data_line[0]
                continue
            if data_line[i] < w_threshold: side_check_w.append(0)
            if data_line[i + 1] < h_threshold: side_check_h.append(0)
            if data_line[i] > col - w_threshold: side_check_w.append(1)
            if data_line[i + 1] > row - h_threshold: side_check_h.append(1)
        sum_w, sum_h = sum(side_check_w), sum(side_check_h)
        if (0 < sum_w < 4) or (0 < sum_h < 4):
            flag = 1
            crease_conf += data_line[-1]

    if crease_length > min(row, col) * 0.1:
        mark = 1
    if crease_length > min(row, col) * 0.5 and flag == 1 and crease_conf > 0.1:
        mark = 2

    return mark

# 现data_line: (conf, *poly, cls)
def score_crease(img_name, save_box_single_dir, img_path):
    original_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    row, col, _ = original_img.shape

    mark = 0
    flag = 0
    crease_length = 0
    crease_conf = 0

    # 阈值选取
    w_threshold, h_threshold = row * 0.2, col * 0.2
    print("score_crease img_name=", img_name)
    path_crease_txt = os.path.join(save_box_single_dir, img_name[:-4] + "__crease.txt")
    if not os.path.exists(path_crease_txt):
        print("something is wrong with path_crease_txt", path_crease_txt)
        return 0

    with open(path_crease_txt, "r", encoding="UTF-8") as fr:
        lines = fr.readlines()

    if not type(lines) == list:
        return 0

    detect_num = len(lines)
    if detect_num == 0:
        return 0
    # if detect_num > 5:
    #     return 2

    for line in lines:
        line_sp = line.strip().split(" ")
        data_line = [float(s_data) for s_data in line_sp]

        # index       0     1    2    3    4    5    6    7    8    9
        # data_line [conf, px1, py1, px2, py2, px3, py3, px4, py4, cls]

        if data_line[-1] == 1:
            if data_line[0] > 0.3:
                mark = 1
            continue
        if data_line[0] < 0.05:
            continue

        # 求矩形最长边
        side_1 = pow(pow(data_line[3] - data_line[1], 2) + pow(data_line[4] - data_line[2], 2), 0.5)
        side_2 = pow(pow(data_line[5] - data_line[1], 2) + pow(data_line[6] - data_line[2], 2), 0.5)
        max_side = max(side_2, side_1)
        crease_length += max_side

        side_check_w, side_check_h = [], []
        for i in range(1, 8, 2):
            if (w_threshold < data_line[i] < col - w_threshold) or (h_threshold < data_line[i + 1] < row - h_threshold):
                flag = 1
                # crease_conf += data_line[-1]
                crease_conf += data_line[0]
                continue
            if data_line[i] < w_threshold: side_check_w.append(0)
            if data_line[i + 1] < h_threshold: side_check_h.append(0)
            if data_line[i] > col - w_threshold: side_check_w.append(1)
            if data_line[i + 1] > row - h_threshold: side_check_h.append(1)
        sum_w, sum_h = sum(side_check_w), sum(side_check_h)
        if (0 < sum_w < 4) or (0 < sum_h < 4):
            flag = 1
            crease_conf += data_line[-1]

    if crease_length > min(row, col) * 0.1:
        mark = 1
    if crease_length > min(row, col) * 0.5 and flag == 1 and crease_conf > 0.1:
        mark = 2

    return mark


if __name__ == "__main__":
    # get_mark_ori()
    pass
