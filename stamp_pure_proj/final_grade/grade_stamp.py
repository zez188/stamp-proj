# -*- coding: UTF-8 -*-

import xgboost as xgb
import numpy as np

# 本py中顺序
# index:            0           1          2        3          4          5       6        7            8              9             10
# score_list = ["pat_pos", "ele_damage", "torn", "defect", "abrasion", "stain", "fade", "crease", "perf_damage", "back_yellow", "back_stain"]
# score_list = [图案位置，    票面要素，    撕裂，    缺损，     磨损，      污渍，   褪色，    折痕，       齿孔，      背变色(泛黄)，   背污渍]

# detect_all.py中顺序  注意：正面褪色和背面褪色要合成一个分数
# index:              0          1         2        3        4           5             6            7        8           9             10
final_cls_list = ["abrasion", "defect", "stain", "crease", "torn", "ele_damage", "perf_damage", "pat_pos", "fade", "back_yellow", "back_stain"]

# 通过字典转换顺序,从detect_all.py到grade_stamp.py, fade和back_fade取最大值合并到fade中
# detect_all.py
dict1 = {0: "abrasion", 1: "defect", 2: "stain", 3: "crease", 4: "torn", 5: "ele_damage",
         6: "perf_damage", 7: "pat_pos", 8: "fade", 9: "back_yellow", 10: "back_stain"}
# grade_stamp.py
dict2 = {"pat_pos": 0, "ele_damage": 1, "torn": 2, "defect": 3, "abrasion": 4, "stain": 5,
         "fade": 6, "crease": 7, "perf_damage": 8, "back_yellow": 9, "back_stain": 10}
dict1_rev = {v: k for k, v in dict1.items()}
dict2_rev = {v: k for k, v in dict2.items()}


def grade_total_score(score_list_str):
    print("score_list_str", score_list_str)
    score_list_num = [int(item.split(":")[-1]) for item in score_list_str]
    print("score_list_num", score_list_num)

    # 转换顺序
    score_list_new = [score_list_num[dict2[dict1[i]]] for i in range(len(score_list_num))]
    print("转换顺序后，未执行finetune_012()前 score_list_new", score_list_new)

    score_list_new = finetune_012(score_list_new)
    pred = final_grade(score_list_new)
    final_score = pred + 45
    grade = map_grade(final_score)

    # 再转换回去
    score_list_new_rev = [score_list_new[dict1_rev[dict2_rev[i]]] for i in range(len(score_list_new))]

    print("打分：", final_score, type(final_score))
    print("等级：", grade, type(grade))
    print("加扣后各项：", score_list_new_rev, type(score_list_new_rev))

    score_list_final_str = [final_cls_list[i] + ":" + str(score_list_new_rev[i]) for i in
                            range(len(score_list_new_rev))]
    print("score_list_final_str", score_list_final_str)
    return int(final_score), grade, score_list_final_str


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
    else:
        # 齿孔加扣磨损
        score_list[4] = score_list[8] if score_list[4] < score_list[8] else score_list[4]

    return score_list


def final_grade(score_list):
    # load model
    model = xgb.Booster()
    model.load_model('final_grade/xgb_database.model')
    # model.load_model('xgb_database.model')

    # make data
    label = np.empty(1)
    # score_list=np.array(score_list)
    # leaf = []
    # leaf.append(score_list)
    leaf = [score_list]
    print("leaf:", leaf)
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
    score_list = ['abrasion:0', 'defect:0', 'stain:0', 'crease:0', 'torn:0', 'ele_damage:2',
                  'perf_damage:2', 'pat_pos:0', 'fade:0', 'back_yellow:0', 'back_stain:0']
    grade_total_score(score_list)
