# -*- coding: UTF-8 -*-


import os
import sys

import joblib

def score_ele_dmg_new(inputs):
    x_input = [inputs]
    print("x_input", x_input)
    score_ele_model_path_list = [r"yolov5_6_1/weights/score_ele_new_11.model",
                                 r"yolov5_6_1/weights/score_ele_new_22.model",
                                 r"yolov5_6_1/weights/score_ele_new_33.model"]
    y_output = 0
    for score_model_path in score_ele_model_path_list:
        if not os.path.exists(score_model_path):
            print("score_ele_model_path does not exist!!!")
            assert False
            # return 0
        clf = joblib.load(score_model_path)
        tmp_y_pred = clf.predict(x_input)
        print(score_model_path + "tmp_y_pred", tmp_y_pred, type(tmp_y_pred))
        y_output += float(tmp_y_pred[0])
    y_output = round(y_output/len(score_ele_model_path_list))
    if y_output > 2:
        y_output = 2
    if y_output < 0:
        y_output = 0
    return y_output


def score_ele_dmg(intersec_area):
    x_input = [intersec_area]
    score_ele_model_path = r"yolov5_6_1/weights/ele_score_pred.model"
    # x_input.append(intersec_area)
    # x_input = intersec_area
    print("x_input", x_input)
    if not os.path.exists(score_ele_model_path):
        print("score_ele_model_path does not exist!!!")
        return 0
    clf = joblib.load(score_ele_model_path)
    y_output = clf.predict(x_input)
    print("y_output", y_output)
    return int(y_output[0])






