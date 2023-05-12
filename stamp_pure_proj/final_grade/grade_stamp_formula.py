# -*- coding: UTF-8 -*-
'''
按照公式计算分数
输入：向量，score_list=[图案，票面要素，撕裂，缺损，磨损，污渍，褪色，折痕，反射效果变暗，齿孔，背变色，背污渍]
输出：整数
0分单项：0
1分单项：-0.4
2分单项：-1
权重：
票面	fA1=60
齿孔	fA2=15
背胶	fA3=15
纸张	fA4=10---无张，暂时按满分处理
分母：
票面 9
齿孔 1
背胶 2
'''


# index:              0          1         2        3        4           5             6            7          8        9          10             11
final_show_list = ["abrasion", "defect", "stain", "crease", "torn", "ele_damage", "perf_damage", "reflect", "pat_pos", "fade", "back_yellow", "back_stain"]

# index:              0           1          2        3          4          5       6        7          8            9             10             11
final_run_list = ["pat_pos", "ele_damage", "torn", "defect", "abrasion", "stain", "fade", "crease", "reflect", "perf_damage", "back_yellow", "back_stain"]
# score_list = [图案位置，    票面要素，    撕裂，    缺损，     磨损，      污渍，   褪色，    折痕，  反射效果变暗，    齿孔，      背变色(泛黄)，    背污渍]

ind_list = [i for i in range(len(final_show_list))]
dict1 = dict(zip(ind_list, final_show_list))
dict2 = dict(zip(final_run_list, ind_list))
dict1_rev = {v: k for k, v in dict1.items()}
dict2_rev = {v: k for k, v in dict2.items()}

def cal_final_with_formula(score_list_str):
    score_list_num_show = [int(item.split(":")[-1]) for item in score_list_str]
    # 转换顺序
    score_list_num_run = [score_list_num_show[dict2[dict1[i]]] for i in range(len(final_show_list))]

    cal_final_score = cal_final_formula(score_list_num_run)

    return score_list_num_show, round(cal_final_score)


def cal_final_formula(score_list):
    SCOREMAP = {0: 0, 1: 0.4, 2: 1}
    dpiaomian = 0
    dchikong = 0
    dbeijiao = 0
    zhizhang = 10
    for i in range(9):
        dpiaomian += SCOREMAP[score_list[i]]
    dchikong += SCOREMAP[score_list[9]]
    print("dchikong", dchikong)
    for i in range(2):
        dbeijiao += SCOREMAP[score_list[i + 10]]
    piaomian = 100 - dpiaomian / 9 * 100
    print("piaomian", piaomian)
    chikong = 100 - dchikong * 100
    print("chikong", chikong)
    beijiao = 100 - dbeijiao / 2 * 100
    print("beijiao", beijiao)
    score = 0.6 * piaomian + 0.15 * chikong + 0.15 * beijiao + zhizhang
    return score


if __name__ == '__main__':
    # score_list = [图案位置,票面要素,撕裂,缺损,磨损,污渍,褪色,折痕,反射效果变暗,齿孔,背变色(泛黄),背污渍]
    score_list = [1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1]
    score = cal_final_formula(score_list)
    print("score", score)
