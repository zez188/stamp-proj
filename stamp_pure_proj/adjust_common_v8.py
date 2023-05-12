import cv2
import math
import numpy as np
from numpy import array
import os

extend_rate = 0.05


def adjust_single_img(file_path):
    adjusted_save_dir = r"save_common_adjusted"
    if not os.path.exists(adjusted_save_dir):
        os.mkdir(adjusted_save_dir)

    original_img, gray_img, RedThresh, closed, opened = Img_Outline(file_path)
    box, draw_img, rect = findContours_img(original_img, opened)
    result_img, mat = Perspective_transform(box, original_img, rect)
    file_name = os.path.basename(file_path)
    adjusted_save_path = os.path.join(adjusted_save_dir, file_name)
    cv2.imencode('.jpg', result_img)[1].tofile(adjusted_save_path)

    # 返回绝对路径
    return os.path.abspath(adjusted_save_path), mat


def adjust_perf_img(file_path):
    adjusted_save_dir = r"save_perf_adjusted"
    if not os.path.exists(adjusted_save_dir):
        os.mkdir(adjusted_save_dir)

    original_img, gray_img, RedThresh, closed, opened = Img_Outline(file_path)
    box, draw_img, rect = findContours_img(original_img, opened)
    result_img, mat = Perspective_transform(box, original_img, rect, is_perf=True)
    file_name = os.path.basename(file_path)
    adjusted_save_path = os.path.join(adjusted_save_dir, file_name)
    cv2.imencode('.jpg', result_img)[1].tofile(adjusted_save_path)

    # 返回绝对路径
    return os.path.abspath(adjusted_save_path), mat


def Img_Outline(input_dir):
    # original_img = cv2.imread(input_dir)
    # original_img=cv2.resize(original_img_01,(600,600),interpolation=cv2.INTER_CUBIC)
    original_img = cv2.imdecode(np.fromfile(input_dir, dtype=np.uint8), -1)
    # original_img=cv2.resize(original_img,(600,800),interpolation=cv2.INTER_CUBIC).astype(np.uint8)

    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_img, (9, 9), 0)  # 高斯模糊去噪
    # _, RedThresh = cv2.threshold(blurred, 165, 255, cv2.THRESH_BINARY)  # 设定阈值120

    # 改
    _, RedThresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))  # 定义矩形结构元素
    closed = cv2.morphologyEx(RedThresh, cv2.MORPH_CLOSE, kernel)  # 闭运算（链接块）
    opened = cv2.morphologyEx(RedThresh, cv2.MORPH_OPEN, kernel, 10)  # 开运算（去噪点）

    return original_img, gray_img, RedThresh, closed, opened


# 找到校正后轮廓
def findContours_img(original_img, opened):
    contours, hierarchy = cv2.findContours(opened, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 改
    h, w = original_img.shape[:2]
    total_area = h * w

    c_list = sorted(contours, key=cv2.contourArea, reverse=True)  # 计算最大轮廓的旋转包围盒
    c_arr = np.empty(shape=(0, 1, 2), dtype=np.float32)
    c_arr_count = 0
    for c in c_list:
        # print("c.shape", c.shape)
        tmp_area = cv2.contourArea(c)

        # 设阈值
        if tmp_area / total_area > 1 / 30:
            c_arr = np.concatenate((c_arr, c), axis=0)
            c_arr_count += 1

    c_arr = c_arr.astype(np.float32)
    rect = cv2.minAreaRect(c_arr)  # 获取包围盒（中心点，宽高，旋转角度）

    box = np.int0(cv2.boxPoints(rect))  # box
    draw_img = cv2.drawContours(original_img.copy(), [box], -1, (0, 0, 255), 1)
    # print("rect:", rect)
    # print("box[0]:", box[0])   # 左下
    # print("box[1]:", box[1])   # 左上
    # print("box[2]:", box[2])   # 右上
    # print("box[3]:", box[3])   # 右下
    return box, draw_img, rect


# 透视变换
def Perspective_transform(box, original_img, rect, is_perf=False):
    if rect[2] > 45:
        pts1 = np.float32([box[0], box[1], box[2], box[3]])
        cutted_W = math.ceil(np.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2))
        cutted_H = math.ceil(np.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2))

    else:
        pts1 = np.float32([box[1], box[2], box[3], box[0]])
        cutted_W = math.ceil(np.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2))
        cutted_H = math.ceil(np.sqrt((box[0][1] - box[1][1]) ** 2 + (box[0][0] - box[1][0]) ** 2))

    ori_h, ori_w = original_img.shape[:2]
    cut_min = min(cutted_W, cutted_H)
    extend_len = int(cut_min * extend_rate)


    if is_perf:
        # 最终确定图片向四周适当延申的长度
        pts2 = np.float32([[extend_len, extend_len],
                           [int(cutted_W + extend_len), extend_len],
                           [int(cutted_W + extend_len), int(cutted_H + extend_len)],
                           [extend_len, int(cutted_H + extend_len)]])

        # 生成透视变换矩阵；进行透视变换
        M = cv2.getPerspectiveTransform(pts1, pts2)

        """
        # cv2.BORDER_WRAP 镜像互换方式
        # cv2.BORDER_ISOLATED 使用黑色进行填充(填充0)
        """
        # 如果邮票四周有相对面的两边都很接近图片边缘
        if abs(ori_h - cutted_H) < 20 or abs(ori_w - cutted_W) < 20:
            result_img = cv2.warpPerspective(original_img, M,
                                             (int(cutted_W + extend_len * 2), int(cutted_H + extend_len * 2)),
                                             borderMode=cv2.BORDER_CONSTANT)
        else:
            result_img = cv2.warpPerspective(original_img, M,
                                             (int(cutted_W + extend_len * 2), int(cutted_H + extend_len * 2)),
                                             borderMode=cv2.BORDER_WRAP)
    else:
        pts2 = np.float32(
            [[0, 0], [int(cutted_W + 1), 0], [int(cutted_W + 1), int(cutted_H + 1)], [0, int(cutted_H + 1)]])
        # 生成透视变换矩阵；进行透视变换
        M = cv2.getPerspectiveTransform(pts1, pts2)
        # 可改
        result_img = cv2.warpPerspective(original_img, M, (int(cutted_W + 1), int(cutted_H + 1)))

    return result_img, M


# 坐标转换
def cvt_pos(pos, cvt_mat):
    """
    u:原x坐标
    v:原y坐标
    x:转换后x坐标
    y:转换后y坐标
    """
    u = pos[0]
    v = pos[1]
    x = (cvt_mat[0][0] * u + cvt_mat[0][1] * v + cvt_mat[0][2]) / (
                cvt_mat[2][0] * u + cvt_mat[2][1] * v + cvt_mat[2][2])
    y = (cvt_mat[1][0] * u + cvt_mat[1][1] * v + cvt_mat[1][2]) / (
                cvt_mat[2][0] * u + cvt_mat[2][1] * v + cvt_mat[2][2])

    new_pos = (int(x), int(y))
    return new_pos


# 主函数，更改路径就行
if __name__ == "__main__":
    file_path = r"2013-25(4-3).jpg"
    common_save_path, mat_com = adjust_single_img(file_path)
    print("common_save_path", common_save_path)

