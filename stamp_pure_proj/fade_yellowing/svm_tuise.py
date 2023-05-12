# import pickle
import numpy as np
import os
import cv2
from PIL import Image

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

import joblib


def contrast(img0):
    """
    计算对比度
    :param img0:图像路径
    :return:对比度数值
    """
    img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    m, n = img1.shape
    img1_ext = cv2.copyMakeBorder(img1, 1, 1, 1, 1, cv2.BORDER_REPLICATE) / 1.0
    rows_ext, cols_ext = img1_ext.shape
    b = 0.0
    for i in range(1, rows_ext - 1):
        for j in range(1, cols_ext - 1):
            b += ((img1_ext[i, j] - img1_ext[i, j + 1]) ** 2 + (img1_ext[i, j] - img1_ext[i, j - 1]) ** 2 +
                  (img1_ext[i, j] - img1_ext[i + 1, j]) ** 2 + (img1_ext[i, j] - img1_ext[i - 1, j]) ** 2)
    cg = b / (4 * (m - 2) * (n - 2) + 3 * (2 * (m - 2) + 2 * (n - 2)) + 2 * 4)
    return cg


def get_feature(img_path):
    """
    计算特征值
    :param img_path:图像路径
    :return:特征列表
    """
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (640, 640))
    b, g, r = cv2.split(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    b_mean = str(np.mean(b)) + ' '
    g_mean = str(np.mean(g)) + ' '
    r_mean = str(np.mean(r)) + ' '
    h_mean = str(np.mean(h)) + ' '
    s_mean = str(np.mean(s)) + ' '
    v_mean = str(np.mean(v)) + ' '
    b_std = str(np.std(b)) + ' '
    g_std = str(np.std(g)) + ' '
    r_std = str(np.std(r)) + ' '
    h_std = str(np.std(h)) + ' '
    s_std = str(np.std(s)) + ' '
    v_std = str(np.std(v)) + ' '
    b_skewness = np.mean(abs(b - b.mean()) ** 3)
    g_skewness = np.mean(abs(g - g.mean()) ** 3)
    r_skewness = np.mean(abs(r - r.mean()) ** 3)
    b_thirdMoment = str(b_skewness ** (1. / 3)) + ' '
    g_thirdMoment = str(g_skewness ** (1. / 3)) + ' '
    r_thirdMoment = str(r_skewness ** (1. / 3)) + ' '
    h_skewness = np.mean(abs(h - h.mean()) ** 3)
    s_skewness = np.mean(abs(s - s.mean()) ** 3)
    v_skewness = np.mean(abs(v - v.mean()) ** 3)
    h_thirdMoment = str(h_skewness ** (1. / 3)) + ' '
    s_thirdMoment = str(s_skewness ** (1. / 3)) + ' '
    v_thirdMoment = str(v_skewness ** (1. / 3)) + ' '


    image = Image.open(img_path)
    small_image = image.resize((640, 640))
    result = small_image.convert('P', palette=Image.ADAPTIVE, colors=5)

    result = result.convert('RGB')
    main_colors = result.getcolors(640 * 640)

    feature = []
    for count, col in main_colors:
        if count < 40:
            continue

        feature.extend(list(col))
    t = [i for i in feature]
    con = contrast(img)
    con = str(con) + ' '
    f = [b_mean, g_mean, r_mean, h_mean, s_mean, v_mean, b_std, g_std, r_std, h_std, s_std, v_std,
         b_thirdMoment, g_thirdMoment, r_thirdMoment, h_thirdMoment, s_thirdMoment, v_thirdMoment, con]
    f.extend(t)
    return f


def tuise(img_path):
    """
    预测邮票是否褪色
    :param img_path:图像路径
    :return:预测结果
    """
    model_path = r'fade_yellowing/svm_tuise_0_58.model'
    # model_path = r'svm_tuise_0_58.model'

    # f = open(model_path, 'rb')  # 加载模型
    # s = f.read()
    # model = pickle.loads(s)
    fade_model = joblib.load(model_path)
    x = get_feature(img_path)  # 提取特征
    x = np.array(x)
    x = x.reshape(1, -1)
    pred = fade_model.predict(x)  # 预测结果
    # print("fade pred=", pred)
    # print("pred", int(pred))
    return int(pred[0])


if __name__ == "__main__":
    tuise("559_J5.3-3.jpg")

