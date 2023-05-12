import pickle
import numpy as np
import os
from PIL import Image

# 主色调
def zhusediao(file_path):
    """
    提取泛黄特征
    :param file_path:图像路径
    :return:泛黄特征
    """
    image = Image.open(file_path)
    small_image = image.resize((640, 640))
    result = small_image.convert('P', palette=Image.ADAPTIVE, colors=5)

    result = result.convert('RGB')
    main_colors = result.getcolors(640 * 640)

    feature = []
    for count, col in main_colors:
        if count < 40:
            continue

        feature.extend(list(col))
    return feature

# 泛黄打分
def score_yellowing(file_path):
    """
    预测邮票是否泛黄
    :param file_path:图像路径
    :return:预测结果
    """
    f1 = open('fade_yellowing/svm_back_yellow_01.model', 'rb')  # 加载模型
    f2 = open('fade_yellowing/svm_back_yellow_12.model', 'rb')
    s1 = f1.read()
    s2 = f2.read()
    model1 = pickle.loads(s1)
    model2 = pickle.loads(s2)
    x = zhusediao(file_path)  # 提取特征
    x = np.array(x)
    x = x.reshape(1, -1)
    pred = model1.predict(x)  # 预测结果

    if int(pred) == 1:
        pred = model2.predict(x)

    return int(pred[0])
