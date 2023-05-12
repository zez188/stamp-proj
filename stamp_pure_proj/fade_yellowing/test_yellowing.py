import pickle
import numpy as np
import os
from PIL import Image


def zhusediao(data_dir):
    """
    提取泛黄特征
    :param data_dir:图像路径
    :return:泛黄特征
    """
    image = Image.open(data_dir)
    small_image = image.resize((640, 640))
    result = small_image.convert('P', palette=Image.ADAPTIVE, colors=5)

    result = result.convert('RGB')
    main_colors = result.getcolors(640 * 640)

    feature = []
    for count, col in main_colors:
        if count < 40:
            continue

        a = np.zeros((224, 224, 3))
        a = a + np.array(col)
        # print(a, col)
        feature.extend(list(col))
    return feature


def fanhuang(data_dir):
    """
    预测邮票是否泛黄
    :param data_dir:图像路径
    :return:预测结果
    """
    f1 = open('svm_fanhuang.model', 'rb')  # 加载模型
    f2 = open('svm_backyellow_12.model', 'rb')
    s1 = f1.read()
    s2 = f2.read()
    model1 = pickle.loads(s1)
    model2 = pickle.loads(s2)
    x = zhusediao(data_dir)  # 提取特征
    x = np.array(x)
    x = x.reshape(1, -1)
    pred = model1.predict(x)  # 预测结果
    print(pred)
    if int(pred) == 1:
        pred = model2.predict(x)
    return pred


if __name__ == '__main__':
    path = r'E:\BaiduNetdiskDownload\stamp\task\fanhuang0519\back_yellow\1'
    data_names = os.listdir(path)
    result = []
    for data_name in data_names:
        data_dir = os.path.join(path, data_name)
        pred = fanhuang(data_dir)
        result.append(int(pred))
    print(result)
