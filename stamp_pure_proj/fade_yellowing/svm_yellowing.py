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

        # a = np.zeros((224, 224, 3))
        # a = a + np.array(col)
        # print(a, col)
        feature.extend(list(col))
    return feature

# 泛黄打分
def score_yellowing(file_path):
    """
    预测邮票是否泛黄
    :param file_path:图像路径
    :return:预测结果
    """
    f = open('fade_yellowing/svm_yellowing_new.model', 'rb')  # 加载模型
    s = f.read()
    model = pickle.loads(s)
    x = zhusediao(file_path)  # 提取特征
    x = np.array(x)
    x = x.reshape(1, -1)
    pred = model.predict(x)  # 预测结果
    return int(pred[0])
