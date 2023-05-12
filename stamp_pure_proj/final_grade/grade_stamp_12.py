# -*- coding: UTF-8 -*-

import os
import xgboost as xgb
import numpy as np

# 1.detect_all.py中顺序 
# index:              0          1         2        3        4           5             6            7          8        9          10             11
final_show_list = ["abrasion", "defect", "stain", "crease", "torn", "ele_damage", "perf_damage", "reflect", "pat_pos", "fade", "back_yellow", "back_stain"]

# 2.grade_stamp_12.py中顺序
# score_list=[图案，票面要素，撕裂，缺损，磨损，污渍，褪色，折痕，反射效果变暗，齿孔，背变色，背污渍]
# index:              0           1          2        3          4          5       6        7          8            9             10             11
final_run_list = ["pat_pos", "ele_damage", "torn", "defect", "abrasion", "stain", "fade", "crease", "reflect", "perf_damage", "back_yellow", "back_stain"]
# score_list = [图案位置，    票面要素，    撕裂，    缺损，     磨损，      污渍，   褪色，    折痕，  反射效果变暗，    齿孔，      背变色(泛黄)，    背污渍]

# assert len(final_show_list) == len(final_run_list)
ind_list = [i for i in range(len(final_show_list))]

# 通过字典转换顺序,从detect_all.py到grade_stamp.py, fade和back_fade取最大值合并到fade中

# detect_all.py
dict1 = dict(zip(ind_list, final_show_list))
print("dict1", dict1)
# dict1 = {0: "abrasion", 1: "defect", 2: "stain", 3: "crease", 4: "torn", 5: "ele_damage",
#          6: "perf_damage", 7: "reflect", 8: "pat_pos", 9: "fade", 10: "back_yellow", 11: "back_stain"}

# grade_stamp_12.py
dict2 = dict(zip(final_run_list, ind_list))
print("dict2", dict2)
# dict2 = {"pat_pos": 0, "ele_damage": 1, "torn": 2, "defect": 3, "abrasion": 4, "stain": 5,
#          "fade": 6, "crease": 7, "reflect": 8, "perf_damage": 9, "back_yellow": 10, "back_stain": 11}

dict1_rev = {v: k for k, v in dict1.items()}
dict2_rev = {v: k for k, v in dict2.items()}


def grade_total_score(score_list_str):

    score_list_num_show = [int(item.split(":")[-1]) for item in score_list_str]
    # print("score_list_num_show", score_list_num_show)

    # 转换顺序
    score_list_num_run = [score_list_num_show[dict1_rev[dict2_rev[i]]] for i in range(len(final_show_list))]
    # print("转换顺序后，未执行finetune_012()前 score_list_num_new", score_list_num_new)

    # 加扣后分数列表score_list_new
    score_list_num_new = finetune_012(score_list_num_run)
    pred = final_grade(score_list_num_new)
    # print("pred", pred, type(pred))
    final_score = int(pred) + 45
    grade = map_grade(final_score)

    # 再转换回去
    score_list_new_rev = [score_list_num_new[dict2[dict1[i]]] for i in range(len(final_show_list))]

    # print("打分：", final_score, type(final_score))
    # print("等级：", grade, type(grade))
    # print("加扣后各项：", score_list_new_rev, type(score_list_new_rev))

    score_list_final_str = [final_show_list[i] + ":" + str(score_list_new_rev[i]) for i in
                            range(len(score_list_new_rev))]
    # print("score_list_final_str", score_list_final_str)
    return final_score, grade, score_list_final_str


def finetune_012(score_list):
    '''加扣'''
    if score_list[3]:
        # 缺损加扣磨损，折痕，断裂
        score_list[2] = score_list[3] if score_list[2] < score_list[3] else score_list[2]
        score_list[4] = score_list[3] if score_list[4] < score_list[3] else score_list[4]
        score_list[7] = score_list[3] if score_list[7] < score_list[3] else score_list[7]
    elif score_list[4]:
        # 磨损加扣缺损，折痕
        score_list[3] = score_list[4] if score_list[3] < score_list[4] else score_list[3]
        score_list[7] = score_list[4] if score_list[7] < score_list[4] else score_list[7]
    elif score_list[2]:
        # 撕裂加扣磨损、折痕、缺损
        score_list[4] = score_list[2] if score_list[4] < score_list[2] else score_list[4]
        score_list[7] = score_list[2] if score_list[7] < score_list[2] else score_list[7]
        score_list[3] = score_list[2] if score_list[3] < score_list[2] else score_list[3]
    elif score_list[9]:
        # 齿孔加扣磨损
        score_list[4] = score_list[9] if score_list[4] < score_list[9] else score_list[4]

    return score_list

def final_grade(score_list):
    # load model
    model = xgb.Booster()
    model_path = r'final_grade/xgb_database_12.model'
    # model_path = "xgb_database_12.model"
    if not os.path.exists(model_path):
        assert os.path.exists(model_path)

    model.load_model(model_path)

    label = np.empty(1)

    leaf = [score_list]

    # print("leaf", leaf)
    dtest = xgb.DMatrix(leaf, label=label)
    pred = model.predict(dtest)
    return pred


# 按照等级匹配
def map_grade(score):
    if score >= 98:
        grade_i = "十级"
    elif score >= 90:
        grade_i = "九级"
    elif score >= 80:
        grade_i = "八级"
    elif score >= 70:
        grade_i = "七级"
    elif score >= 60:
        grade_i = "六级"
    elif score >= 50:
        grade_i = "五级"
    else:
        grade_i = "四级"
    return grade_i


if __name__ == '__main__':
    # final_show_list = []
    # score_list = [score_list_num_show[dict2[dict1[i]]] for i in range(len(final_show_list))]
    score_list = [2, 2, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0]
    finetune_012(score_list)
    pred = final_grade(score_list)
    final_score = pred + 45
    grade=map_grade(final_score)
    print("打分：",final_score)
    print("等级：",grade)
    print("加扣后各项：",score_list)

