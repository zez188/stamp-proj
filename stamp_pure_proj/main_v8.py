import sys
import os
import shutil

sys.path.append(os.path.abspath("yolov5_6_1"))
sys.path.append(os.path.abspath("yolov5_obb_crease"))
sys.path.append(os.path.abspath("QtProject"))
# print(sys.path)

import time
import random
import json
import numpy as np
from QtProject.ui.MainWindow_v5 import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal

from QtProject import predict_0720
from detect_all_v8 import detectFront, detectBack
from child_window import Child

from adjust_common_v8 import adjust_single_img, adjust_perf_img

from pattern_pos.pat_pos_adjust import adjust_pat_img
# from adjust_perf import adjust_perf_img
import cut_margin

from final_grade.grade_stamp_12 import grade_total_score

# 改stb_path
stb_path = os.path.abspath('/home/zez/dataset-downloads/standardlib')
assert os.path.exists(stb_path), "标准库导入路径错误"

class Timer(QThread):
    """创建一个计时器类"""
    sinOut = pyqtSignal(int, bool)

    # 初始化一些可能会用到的变量
    def __init__(self):
        super().__init__()
        self.is_shown = False
        self.val = 0

    def setVal(self, is_shown):
        self.is_shown = is_shown

        # 执行线程的run方法
        self.start()

    # 重写QThread的run函数
    def run(self):
        print("timer is running")
        print("self.is_shown", self.is_shown)
        while self.is_shown:
            # 发射信号
            self.val += int(random.random() * random.randint(15, 30))
            print("self.val", self.val)
            self.sinOut[int, bool].emit(min(self.val, 99), True)
            time.sleep(random.uniform(1.0, 1.3))
        else:
            self.val = 0
            self.sinOut[int, bool].emit(0, False)


# 警告框
def warn_dialog(content="error"):
    msg_box = QMessageBox(QMessageBox.Warning, 'Warning', content)
    msg_box.exec_()


class MainWindow(QMainWindow):
    cls_signal = pyqtSignal(str)
    label_score_signal = pyqtSignal([str, str, str, str, str, str, object], [str, str, str, str, str, str, object, str])
    cvt_mat_signal = pyqtSignal(object)

    def __init__(self):
        super(MainWindow, self).__init__()

        bg1_pic_abs_path = os.path.abspath("QtProject/ui/logo_1.svg").replace("\\", "/")
        bg2_pic_abs_path = os.path.abspath("QtProject/ui/view_2.jpg").replace("\\", "/")
        # print("bg1_pic_abs_path", bg1_pic_abs_path)
        # print("bg2_pic_abs_path", bg2_pic_abs_path)

        # 分类模块识别出的邮票标准命名，后缀固定.jpg
        self.cls_name = None

        # 输入邮票图片的正面\背面路径
        self.file_front_path = None
        self.file_back_path = None

        self.file_front_comadj_path = None
        self.file_front_posadj_path = None
        self.file_front_peradj_path = None
        self.file_perf_dir = None
        self.file_back_comadj_path = None

        self.mat_com = None
        # self.mat_per = None

        # progressBar label_9 label_10 的当前显示/隐藏状态
        self.last_state = False

        # 正面\背面子窗口
        self.child_front = None
        self.child_back = None

        # 输出已标记的邮票图片的正面\背面的保存路径、打分列表
        self.labeled_front_path = None
        self.score_front_list = None
        self.labeled_back_path = None
        self.score_back_list = None

        # 初始化ui界面
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self, bg1_abs_path=bg1_pic_abs_path, bg2_abs_path=bg2_pic_abs_path)
        self.ui.label_9.setVisible(False)
        self.ui.label_10.setVisible(False)
        self.ui.progressBar.setVisible(False)

        self.thread_cls = QThread()
        self.worker_cls = Worker()
        self.worker_cls.moveToThread(self.thread_cls)
        self.worker_cls.cls_result_signal[str, float].connect(self.show_cls_result)
        self.cls_signal[str].connect(self.worker_cls.cls)

        self.thread_score = QThread()
        self.worker_label_score = Worker()
        self.worker_label_score.moveToThread(self.thread_score)

        self.thread_time = Timer()  # Timer()继承QThread()
        self.thread_time.sinOut[int, bool].connect(self.show_progress)

        self.worker_label_score.label_score_result_signal[str, list, str].connect(self.show_label_score_result)
        self.worker_label_score.final_score_result_signal[int, str, list].connect(self.show_final_score)
        self.worker_label_score.progressBar_signal[bool].connect(self.thread_time.setVal)

        self.label_score_signal[str, str, str, str, str, str, object].connect(self.worker_label_score.label_score)
        self.label_score_signal[str, str, str, str, str, str, object, str].connect(self.worker_label_score.label_score)
        # self.cvt_mat_signal[object].connect(self.worker_label_score.get_cvt_mat)

    # 重置所有
    def reset_all(self):

        self.del_pics_maked(self.file_front_comadj_path,
                            self.file_front_peradj_path,
                            self.file_perf_dir,
                            self.file_front_posadj_path,
                            self.file_back_comadj_path)  # save_anchored_img_dir不删除

        self.cls_name = None

        self.file_front_path = None
        self.file_back_path = None

        self.file_front_comadj_path = None
        self.file_front_posadj_path = None
        self.file_front_peradj_path = None
        self.file_perf_dir = None
        self.file_back_comadj_path = None

        self.mat_com = None
        # self.mat_per = None

        self.last_state = False

        self.child_front = None
        self.child_back = None

        self.labeled_front_path = None
        self.score_front_list = None
        self.labeled_back_path = None
        self.score_back_list = None

        self.ui.progressBar.setVisible(False)
        self.ui.label_9.setVisible(False)
        self.ui.label_10.setVisible(False)

        self.ui.label.setText("正面图片")
        self.ui.label_2.setText("背面图片")
        self.ui.label_3.setText("分类结果")

        self.ui.label_4.clear()
        self.ui.label_4.setText("未选择图片")
        self.ui.label_5.clear()
        self.ui.label_5.setText("未选择图片")
        self.ui.label_6.clear()
        self.ui.label_6.setText("无分类结果")

        self.ui.label_7.setText("打分结果：无")
        self.ui.label_8.setText("划分等级：无")

        self.ui.label_9.clear()
        self.ui.label_10.clear()

    # 删掉程序运行生成的所有图片，除了save_anchored_img_dir里的已标注图片
    def del_pics_maked(self, *paths):
        # print("paths", paths)
        for tmp_path in paths:
            if tmp_path and os.path.exists(tmp_path):
                if os.path.isfile(tmp_path):
                    os.remove(tmp_path)
                elif os.path.isdir(tmp_path):
                    shutil.rmtree(tmp_path)  # 递归删除非空文件夹
        print("reset_all() all pics_maked have benn deleted over")

    # 选择图片
    def slc_msg(self, side: str):
        if side == "front":
            # self.file_front_path, filetype = QtWidgets.QFileDialog.getOpenFileName(None, '选择图片', img_path, '*jpg')
            # self.file_front_path, _ = QtWidgets.QFileDialog.getOpenFileName(None, '选择图片', 'D:/', '*jpg')
            self.file_front_path, _ = QtWidgets.QFileDialog.getOpenFileName(None, '选择图片', '.', '*jpg')

            print("self.file_front_path", self.file_front_path)
            if not os.path.isfile(self.file_front_path):
                self.file_front_path = None
                return
            if not os.access(self.file_front_path, os.R_OK):
                file_name = os.path.basename(self.file_front_path)
                warn_content = "当前用户没有读取 " + file_name + " 的权限"
                print(warn_content)
                warn_dialog(warn_content)
                return
            self.file_front_path = os.path.abspath(self.file_front_path).replace("\\", "/")
            self.ui.label_6.setText('无分类结果')

            if self.file_front_path:

                self.file_front_comadj_path, self.mat_com = adjust_single_img(self.file_front_path)
                self.file_front_comadj_path = self.file_front_comadj_path.replace("\\", "/")

                self.file_front_peradj_path, _ = adjust_perf_img(self.file_front_path)
                self.file_front_peradj_path = self.file_front_peradj_path.replace("\\", "/")
                self.file_perf_dir = cut_margin.cut_perf_adj_margin(self.file_front_comadj_path,
                                                                    self.file_front_peradj_path)
                self.file_perf_dir = self.file_perf_dir.replace("\\", "/")

                self.file_front_posadj_path = adjust_pat_img(self.file_front_path)
                self.file_front_posadj_path = self.file_front_posadj_path.replace("\\", "/")

                jpg_img = QtGui.QPixmap(self.file_front_comadj_path).scaled(self.ui.label_4.width(),
                                                                            self.ui.label_4.height(),
                                                                            Qt.KeepAspectRatio,
                                                                            Qt.SmoothTransformation)
                self.ui.label_4.setPixmap(jpg_img)
            else:
                self.ui.label_4.setText('未选择正面图片')
        elif side == "back":
            # self.file_back_path, filetype = QtWidgets.QFileDialog.getOpenFileName(None, '选择图片', img_path, '*jpg')
            self.file_back_path, _ = QtWidgets.QFileDialog.getOpenFileName(None, '选择图片', '.', '*jpg')

            print("self.file_back_path", self.file_back_path)
            if not os.path.isfile(self.file_back_path):
                self.file_back_path = None
                return
            if not os.access(self.file_back_path, os.R_OK):
                file_name = os.path.basename(self.file_back_path)
                warn_content = "当前用户没有读取 " + file_name + " 的权限"
                print(warn_content)
                warn_dialog(warn_content)
                return
            self.file_back_path = self.file_back_path.replace("\\", "/")
            self.ui.label_6.setText('无分类结果')
            # self.ui.label_6.setText('')
            if self.file_back_path:
                self.file_back_comadj_path, _ = adjust_single_img(self.file_back_path)
                self.file_back_comadj_path = self.file_back_comadj_path.replace("\\", "/")
                jpg = QtGui.QPixmap(self.file_back_comadj_path).scaled(self.ui.label_5.width(),
                                                                       self.ui.label_5.height(),
                                                                       Qt.KeepAspectRatio,
                                                                       Qt.SmoothTransformation
                                                                       )
                self.ui.label_5.setPixmap(jpg)
            else:
                self.ui.label_5.setText('未选择背面图片')
        else:
            print("something is wrong with slc_msg()")
            self.ui.label_4.setText("something is wrong with slc_msg()")
            self.ui.label_5.setText("something is wrong with slc_msg()")

    # 开始执行分类
    def cls_msg(self):
        # 之后可能改成self.file_front_comadj_path
        if self.file_front_path:
            self.thread_cls.start()
            self.ui.label_6.setText('分类中……')

            self.cls_signal[str].emit(self.file_front_path)
            # self.cls_signal[str].emit(self.file_front_comadj_path)
        else:
            self.ui.label_6.setText('未选择图片！！！')

    # 显示分类结果
    def show_cls_result(self, cls_result, cls_prob):
        print("cls_result, cls_prob", cls_result, cls_prob)
        if cls_result == "NoneNoneNone":
            warn_dialog("标准库未找到")
            self.ui.label_6.setText('标准库未找到，无法正确分类')
            return
        self.cls_name = os.path.basename(cls_result)

        cls_prob_thres = 0.85
        if cls_prob > cls_prob_thres and os.path.exists(cls_result):
            if "T46.1-1" in self.cls_name:
                self.ui.label_3.setText("分类结果: " + self.cls_name + "(庚申猴票)")
            else:
                self.ui.label_3.setText("分类结果: " + self.cls_name)
            cls_result_jpg = QtGui.QPixmap(cls_result).scaled(self.ui.label_6.width(),
                                                              self.ui.label_6.height(),
                                                              Qt.KeepAspectRatio,
                                                              Qt.SmoothTransformation
                                                              )
            self.ui.label_6.setPixmap(cls_result_jpg)
        # elif os.path.exists(cls_result):
        #     self.ui.label_3.setText("分类结果: " + "该邮票不在标准库中")
        #     cls_result_jpg = QtGui.QPixmap(cls_result).scaled(self.ui.label_6.width(),
        #                                                       self.ui.label_6.height(),
        #                                                       Qt.KeepAspectRatio,
        #                                                       Qt.SmoothTransformation
        #                                                       )
        #     self.ui.label_6.setPixmap(cls_result_jpg)
        else:
            self.ui.label_3.setText("分类结果: " + "该邮票不在标准库中")
            self.ui.label_6.setText("该邮票在标准库中匹配失败")

    # 开始执行标记、打分，label_score_signal发信号
    def label_and_score(self):
        self.thread_score.start()
        if self.file_front_comadj_path and self.file_back_comadj_path:
            # self.thread_score.start()
            self.ui.label.setText('邮票正面品鉴中……')
            self.ui.label_2.setText('邮票背面品鉴中……')
            self.label_score_signal[str, str, str, str, str, str, object, str].emit(self.file_front_path,
                                                                                    self.file_front_comadj_path,
                                                                                    self.file_front_posadj_path,
                                                                                    self.file_front_peradj_path,
                                                                                    self.file_perf_dir,
                                                                                    self.cls_name,
                                                                                    self.mat_com,
                                                                                    self.file_back_comadj_path)
        else:
            if not self.file_front_comadj_path:
                self.ui.label_4.setText('未选择正面图片！！！')
            if not self.file_back_comadj_path:
                # self.thread_score.start()
                self.ui.label.setText('正面标记、打分中……')
                self.label_score_signal[str, str, str, str, str, str, object].emit(self.file_front_path,
                                                                                   self.file_front_comadj_path,
                                                                                   self.file_front_posadj_path,
                                                                                   self.file_front_peradj_path,
                                                                                   self.file_perf_dir,
                                                                                   self.cls_name,
                                                                                   self.mat_com)
                self.ui.label_5.setText('未选择背面图片！！！')

    # 显示标记、打分的结果
    def show_label_score_result(self, label_res: str, score_res: [str, list], side: [str]):
        label_res_str = str(label_res)
        print("show_label_score_result() label_res=", label_res_str)

        if os.path.exists(label_res):
            print("已完成品鉴, 已保存 " + side + " 邮票")
            if side == "front":
                self.labeled_front_path = label_res_str
                self.score_front_list = score_res
                self.ui.label.setText("正面已标注")
                self.show_child("front")
            if side == "back":
                self.labeled_back_path = label_res_str
                self.score_back_list = score_res
                self.ui.label_2.setText("背面已标注")
                self.show_child("back")
        else:
            show_msg = label_res + " " + side + "does not exist!!!"
            # print(show_msg)
            self.ui.label.setText(show_msg)

    # 子界面显示已标注的正面/背面
    def show_child(self, side: str):
        if side == "front":
            if not self.labeled_front_path:
                print("show_child() self.labeled_front_path == None")
                warn_dialog("没有正面标注图")
                return
            self.child_front = Child(self.labeled_front_path, self.score_front_list, side)
            self.child_front.show()
        if side == "back":
            if not self.labeled_back_path:
                print("show_child() self.labeled_back_path == None")
                warn_dialog("没有背面标注图")
                return
            self.child_back = Child(self.labeled_back_path, self.score_back_list, side)
            self.child_back.show()

    def show_final_score(self, final_score: int, grade: str, score_list_all: list):
        # print("show_final_score()")
        self.ui.label_7.setText("打分结果：" + str(final_score))
        self.ui.label_8.setText("划分等级：" + grade)
        front_scores_str = ""
        back_scores_str = ""
        for i in range(len(score_list_all)):
            if i < 10:
                front_scores_str += " " + score_list_all[i]
            else:
                back_scores_str += " " + score_list_all[i]
        self.ui.label_9.setText("加扣后,正面：" + front_scores_str)
        self.ui.label_10.setText("加扣后,背面：" + back_scores_str)

    def show_progress(self, pg_value, is_shown):
        print("self.last_state, is_shown", self.last_state, is_shown)
        if self.last_state != is_shown:
            if self.last_state:  # last_state 由True->False
                self.ui.label_9.setVisible(True)
                self.ui.label_10.setVisible(True)
            else:  # last_state 由False->True
                self.ui.progressBar.setVisible(True)
                self.ui.label_9.setVisible(False)
                self.ui.label_10.setVisible(False)
            self.last_state = is_shown
        if is_shown:
            self.ui.progressBar.setValue(pg_value)
        else:  # isShown == False
            self.ui.progressBar.setVisible(False)
            self.ui.progressBar.setValue(0)


class Worker(QObject):
    cls_result_signal = pyqtSignal(str, float)

    # label_score_result_signal = pyqtSignal([str, str, str], [str, list, str])
    label_score_result_signal = pyqtSignal(str, list, str)
    final_score_result_signal = pyqtSignal(int, str, list)
    progressBar_signal = pyqtSignal(bool)

    # 运行main()分类模型,信号cls_result_signal,发送cls_result
    def cls(self, file_front_path):
        start_time = time.time()
        if not os.path.exists(stb_path):
            cls_result = "NoneNoneNone"
            cls_prob = 0
            self.cls_result_signal[str, float].emit(cls_result, cls_prob)
            lasted_time = time.time() - start_time
            print("cls() totally spend time:", lasted_time)
            return

        cls_name, cls_prob = predict_0720.main(file_front_path)

        json_cls_name_path = r"class_indices.json"
        with open(json_cls_name_path, 'r', encoding='utf8')as fp:
            dict_cls = json.load(fp)

        if not (cls_name in dict_cls):
            cls_name_zh = cls_name + ".jpg"
        else:
            cls_name_zh = dict_cls[cls_name]

        cls_result = os.path.join(stb_path, cls_name_zh)
        self.cls_result_signal[str, float].emit(cls_result, cls_prob)

        lasted_time = time.time() - start_time
        print("cls() totally spend time:", lasted_time)

    # 运行detectItems()包含6个标框模型,信号label_score_result_signal,发送label_score_result
    def label_score(self,
                    file_front_ori_path: [str, None] = None,
                    file_front_comadj_path: [str, None] = None,
                    file_front_posadj_path: [str, None] = None,
                    file_front_peradj_path: [str, None] = None,
                    file_front_perf_dir: [str, None] = None,
                    file_cls_name: [str, None] = None,
                    mat_com: [np.ndarray, None] = None,
                    file_back_comadj_path: [str, None] = None):

        start_time = time.time()

        # 改此处即可切换中文还是英文
        use_zh = True

        if not file_front_comadj_path:
            print("label_score() is wrong, file_front_comadj_path does not exist")
            score_list = [0 for _ in range(12)]
            return score_list

        self.progressBar_signal[bool].emit(True)
        front_labelled_path, front_score_res = detectFront(file_front_ori_path,
                                                           file_front_comadj_path,
                                                           file_front_posadj_path,
                                                           file_front_peradj_path,
                                                           file_front_perf_dir,
                                                           file_cls_name,
                                                           mat_com,
                                                           use_zh)
        # print("front_labelled_path", front_labelled_path)
        # print("front_score_res", front_score_res)

        self.label_score_result_signal[str, list, str].emit(front_labelled_path, front_score_res, "front")

        if not file_back_comadj_path:
            score_list = front_score_res + ["back_yellow:0", "back_stain:0"]
        else:  # file_back_comadj_path does exist
            back_labelled_path, back_score_res = detectBack(file_back_comadj_path, file_cls_name, use_zh)
            # print("back_labelled_path", back_labelled_path)
            # print("back_score_res", back_score_res)

            score_list = front_score_res + back_score_res

            # print("before grade_total_score() score_list", score_list)

            self.label_score_result_signal[str, list, str].emit(back_labelled_path, back_score_res, "back")

        final_score, grade, score_list_all = grade_total_score(score_list_str=score_list)
        # print("main(), final_score, grade, score_list_all", final_score, grade, score_list_all)

        self.final_score_result_signal[int, str, list].emit(final_score, grade, score_list_all)
        self.progressBar_signal[bool].emit(False)

        lasted_time = time.time() - start_time
        print("label_score() totally spend time:", lasted_time)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
