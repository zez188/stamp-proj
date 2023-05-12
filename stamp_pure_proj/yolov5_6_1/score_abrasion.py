# -*- coding: UTF-8 -*-


def grade_abrasion(score_abrasion_inputs):
    file_list, abrasion_list, ratio_list, center_point_list = score_abrasion_inputs
    num = len(file_list)
    score_abrasion_list = []
    print(ratio_list)
    for i in range(0, num):
        if int(abrasion_list[i]) > 0:
            if ratio_list[i] > 0.022:
                score_abrasion_list.append(2)
            else:
                if center_point_list[i] == 1:
                    if ratio_list[i] > 0.009:
                        score_abrasion_list.append(2)
                    else:
                        score_abrasion_list.append(1)
                else:
                    score_abrasion_list.append(1)
        else:
            score_abrasion_list.append(0)
    print("len(score_abrasion_list)", len(score_abrasion_list))
    print("score_abrasion_list", score_abrasion_list)

    return int(score_abrasion_list[0])
