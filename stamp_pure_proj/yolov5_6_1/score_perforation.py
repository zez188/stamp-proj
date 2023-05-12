# -*- coding: UTF-8 -*-
import os
import pandas as pd
import joblib

def score_perf_new(img, label_xyxy_inputs):
    img_h, img_w = img.shape[0], img.shape[1]
    perf_num = 0
    perf_len_max = 0
    # perf_area = 0

    # 每张图切4个边，len(label_xyxy_inputs) == 4
    for i in range(len(label_xyxy_inputs)):
        perf_num += len(label_xyxy_inputs[i])

        for conf, xyxy in label_xyxy_inputs[i]:
            xml_w = float(abs(int(xyxy[2]) - int(xyxy[0])) / img_w)
            xml_h = float(abs(int(xyxy[3]) - int(xyxy[1])) / img_h)
            perf_len_max = max(perf_len_max, xml_w, xml_h)
            # perf_area += xml_w * xml_h

    if perf_num == 0:
        return 0

    x = pd.DataFrame()
    x['len_max'] = [perf_len_max]
    x['num'] = [perf_num]
    # x['area'] = [perf_area]
    print("score_pref_new() x =", x)
    print("os.getcwd() =", os.getcwd())

    # score_perf_model_path1 = r"yolov5_6_1/weights/score_perf_1_0720.model"
    # score_perf_model_path2 = r"yolov5_6_1/weights/score_perf_2_0720.model"
    # score_perf_model_path3 = r"yolov5_6_1/weights/score_perf_3_0720.model"
    score_perf_model_path1 = r"yolov5_6_1/weights/score_perf_new_1.model"
    score_perf_model_path2 = r"yolov5_6_1/weights/score_perf_new_2.model"
    score_perf_model_path3 = r"yolov5_6_1/weights/score_perf_new_3.model"
    score_perf_model_pathList = [score_perf_model_path1, score_perf_model_path2, score_perf_model_path3]

    y_hat_average = 0
    for tmp_model_path in score_perf_model_pathList:
        tmp_model_abs_path = os.path.abspath(tmp_model_path)
        print("tmp_model_abs_path =", tmp_model_abs_path)
        clf = joblib.load(tmp_model_path)
        y_hat = float(clf.predict(x))
        y_hat_average += y_hat

    y_hat_average = round(y_hat_average/len(score_perf_model_pathList))
    print("score_perf() y_hat_average =", y_hat_average)
    return y_hat_average


# [已弃用]
def score_perf(img_name, save_box_single_dir, img_ori):
    img_h, img_w = img_ori.shape[0], img_ori.shape[1]

    perf_num = 0
    perf_len_max = 0

    for i in range(1, 5):
        tmp_txt_path = os.path.join(save_box_single_dir, img_name[:-4]+"_"+str(i)+"__perf.txt")
        with open(tmp_txt_path, "r", encoding='UTF-8') as fr:
            lines = fr.readlines()
            if len(lines) == 0:
                continue
            perf_num += len(lines)
            for line in lines:
                line_sp = line.strip().split(" ")
                xml_w = float(abs(int(line_sp[3])-int(line_sp[1]))/img_w)
                xml_h = float(abs(int(line_sp[4])-int(line_sp[2]))/img_h)
                perf_len_max = max(perf_len_max, xml_w, xml_h)

    if (perf_num == 0) and (perf_len_max == 0):
        return 0
    x = pd.DataFrame()
    x['len_max'] = [perf_len_max]
    x['num'] = [perf_num]
    print("x", x)
    print("os.getcwd() =", os.getcwd())
    score_perf_model_pathList = [r"yolov5_ele/weights/score_perf_1.model",
                                    r"yolov5_ele/weights/score_perf_2.model",
                                    r"yolov5_ele/weights/score_perf_3.model"]
    y_hat_average = 0
    for tmp_model_path in score_perf_model_pathList:
        tmp_model_abs_path = os.path.abspath(tmp_model_path)
        print("tmp_model_abs_path =", tmp_model_abs_path)
        clf = joblib.load(tmp_model_path)
        y_hat = float(clf.predict(x))
        y_hat_average += y_hat

    y_hat_average = round(y_hat_average/len(score_perf_model_pathList))
    print("score_perf() y_hat_average =", y_hat_average)
    return y_hat_average
