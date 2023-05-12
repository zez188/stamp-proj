# -*- coding: UTF-8 -*-

import os
import sys

sys.path.append(os.path.abspath("yolov5_6_1"))
sys.path.append(os.path.abspath("yolov5_obb_crease"))
# print(sys.path)

import xlwt
# from adjust_common_v4 import adjust_single_img
from adjust_common_v8 import adjust_single_img, adjust_perf_img

from pattern_pos.pat_pos_adjust import adjust_pat_img
from adjust_perf import adjust_perf_img
import cut_margin

from detect_all_v8 import detectFront, detectBack

from final_grade.grade_stamp_formula import cal_final_with_formula
from final_grade.grade_stamp_12 import grade_total_score
from QtProject import predict_0720

# index:              0          1         2        3        4           5             6            7          8        9          10             11
final_show_list = ["abrasion", "defect", "stain", "crease", "torn", "ele_damage", "perf_damage", "reflect", "pat_pos", "fade", "back_yellow", "back_stain"]


def final_score_fill(pic_nice_dir):
    pic_list = os.listdir(pic_nice_dir)
    pic_list_sorted = sorted(pic_list)
    print("pic_list_sorted", pic_list_sorted)

    list_file_score = []
    for pic_name in pic_list_sorted:
        if pic_name[-7:] == "(2).jpg":
            continue
        pic_back_name = pic_name[:-4] + "(2).jpg"
        # print(pic_name, pic_back_name)
        file_front_path = os.path.join(pic_nice_dir, pic_name).replace("\\", "/")
        file_back_path = os.path.join(pic_nice_dir, pic_back_name).replace("\\", "/")
        print(file_front_path, file_back_path)

        file_front_comadj_path, mat_com = adjust_single_img(file_front_path)
        file_front_comadj_path = file_front_comadj_path.replace("\\", "/")

        file_front_peradj_path, _ = adjust_perf_img(file_front_path)
        file_front_peradj_path = file_front_peradj_path.replace("\\", "/")
        file_front_perf_dir = cut_margin.cut_perf_adj_margin(file_front_comadj_path, file_front_peradj_path).replace("\\", "/")
        # file_perf_dir = file_perf_dir.replace("\\", "/")

        file_front_posadj_path = adjust_pat_img(file_front_path).replace("\\", "/")
        # file_front_posadj_path = file_front_posadj_path.replace("\\", "/")

        file_cls_name, file_cls_prob = predict_0720.main(file_front_path)

        use_zh = True
        front_labelled_path, front_score_res = detectFront(file_front_path,
                                                           file_front_comadj_path,
                                                           file_front_posadj_path,
                                                           file_front_peradj_path,
                                                           file_front_perf_dir,
                                                           file_cls_name,
                                                           mat_com,
                                                           use_zh)

        if not os.path.exists(file_back_path):
            # file_back_comadj_path = None
            # back_labelled_path = None
            back_score_res = ["back_yellow:0", "back_stain:0"]
        else:
            file_back_comadj_path, _ = adjust_single_img(file_back_path)
            file_back_comadj_path = file_back_comadj_path.replace("\\", "/")

            back_labelled_path, back_score_res = detectBack(file_back_comadj_path, file_cls_name, use_zh)

        score_list = front_score_res + back_score_res
        fit_final_score, _1, _2 = grade_total_score(score_list_str=score_list)
        score_list_num_show, cal_final_score = cal_final_with_formula(score_list_str=score_list)

        # 加权
        # final_score_weighted = fit_final_score * 0.9 + cal_final_score * 0.1
        # final_score_weighted = cal_final_score
        final_score_weighted = fit_final_score

        # dict_file_score[file_front_path] = final_score
        list_file_score.append((pic_name, score_list_num_show, final_score_weighted))

    return list_file_score


if __name__ == "__main__":
    pic_nice_dir = r"test1010"
    pic_nice_dir = os.path.abspath(pic_nice_dir)

    list_file_score = final_score_fill(pic_nice_dir)
    # list_file_score = []
    print("list_file_score", list_file_score)

    #创建一个Workbook对象，相当于创建了一个Excel文件
    book = xlwt.Workbook(encoding="UTF-8")

    # 创建一个sheet对象，一个sheet对象对应Excel文件中的一张表格。
    sheet = book.add_sheet('邮票品鉴', cell_overwrite_ok=True)

    # 填入第一行
    list0 = ["图像名称",
             "磨损", "缺损", "污渍", "折痕", "撕裂", "票面要素", "齿孔", "反射效果变暗", "图案位置", "褪色",
             "背面泛黄", "背面污渍",
             "系统评分"]

    for i in range(len(list0)):
        sheet.write(0, i, list0[i], style=xlwt.easyxf('pattern: pattern solid, fore_colour yellow'))

    # 填入第i列，从每列第二行开始填
    for j in range(len(list_file_score)):
        sheet.write(j + 1, 0, list_file_score[j][0])
        for k in range(12):
            sheet.write(j + 1, k + 1, str(list_file_score[j][1][k]))
        sheet.write(j + 1, 13, str(list_file_score[j][2]))

    # 保存.xls
    book.save('邮票系统测试_北京0930测试集_1101_v7_v4.xls')

