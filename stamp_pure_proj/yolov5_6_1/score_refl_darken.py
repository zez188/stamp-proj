# -*- coding: UTF-8 -*-


def score_refl(c0_xyxy_list, c1_xyxy_list):
    refl_score = 0
    if len(c1_xyxy_list) > 0:
        refl_score = 2
    elif len(c0_xyxy_list) > 0:
        refl_score = 1

    return refl_score