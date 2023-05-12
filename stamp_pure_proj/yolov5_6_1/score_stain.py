# -*- coding: UTF-8 -*-

import os
import numpy as np
import xgboost as xgb


def grade_stain_mould(score_stain_inputs):
    file_list, mould_list, stain_list, ratio_list = score_stain_inputs

    '''xgboost打分'''
    model = xgb.Booster()
    # model.load_model('weights/xgb_back.model')
    model_path = 'yolov5_6_1/weights/xgb_stain.model'
    if not os.path.exists(model_path):
        assert os.path.exists(model_path)
    model.load_model(model_path)

    # make DMatrix
    leaf = list(zip(mould_list, stain_list, ratio_list))
    label = np.zeros(len(file_list))
    # print("leaf", leaf)
    dtest = xgb.DMatrix(leaf, label=label)
    # give a grade
    score_stain_list = model.predict(dtest)
    print("len(score_stain_list)", len(score_stain_list))
    print("score_stain_list", score_stain_list)

    return int(score_stain_list[0])
