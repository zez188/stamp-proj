import sys
import os
from ui.MainWindow import Ui_MainWindow
from predict import main
from detect_all import detectItems, cv_imread, cv_imwrite
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import QObject, QThread, pyqtSignal

img_path = r'E:\BaiduNetdiskDownload\stamp\stamp_classify_adjust'
stb_path = r'E:\BaiduNetdiskDownload\stamp\standardlib_adjust'


class MainWindow(QMainWindow):
    cls_signal = pyqtSignal(str)
    label_score_signal = pyqtSignal(str)

    def __init__(self):
        super(MainWindow, self).__init__()
        self.filename = None
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.thread_cls = QThread()

        self.worker_cls = Worker()
        self.worker_cls.moveToThread(self.thread_cls)
        self.worker_cls.cls_result_signal.connect(self.show_result)

        self.worker_label_score = Worker()
        self.worker_label_score.moveToThread(self.thread_cls)

        self.cls_signal.connect(self.worker_cls.cls)
        self.label_score_signal.connect(self.worker_cls.label_score)

    def slc_msg(self):
        self.filename, filetype = QtWidgets.QFileDialog.getOpenFileName(None, '选择图片', img_path, '*jpg')
        self.ui.label_5.setText('无分类结果')
        self.ui.label_6.setText('')
        if self.filename:
            self.ui.label_3.setText(self.filename)
            jpg = QtGui.QPixmap(self.filename).scaled(self.ui.label_4.width(), self.ui.label_4.height())
            self.ui.label_4.setPixmap(jpg)
        else:
            self.ui.label_4.setText('未选择图片')

    def cls_msg(self):
        if self.filename:
            self.thread_cls.start()
            self.ui.label_5.setText('分类中……')
            self.cls_signal.emit(self.filename)
        else:
            self.ui.label_5.setText('未选择图片！！！')


    def label_and_score(self):
        if self.filename:
            self.thread_cls.start()
            self.ui.label_5.setText('标记、打分中……')
            self.label_score_signal.emit(self.filename)
        else:
            self.ui.label_5.setText('未选择图片！！！')


    def show_result(self, cls_result):
        self.ui.label_6.setText(cls_result.split('\\')[-1])
        if os.path.exists(cls_result):
            cls_result_jpg = QtGui.QPixmap(cls_result).scaled(self.ui.label_5.width(), self.ui.label_5.height())
            self.ui.label_5.setPixmap(cls_result_jpg)
        else:
            self.ui.label_5.setText(cls_result)


class Worker(QObject):
    cls_result_signal = pyqtSignal(str)
    label_score_result_signal = pyqtSignal(str)

    def cls(self, filename):
        clsname, print_res = main(filename)
        cls_result = os.path.join(stb_path, clsname + '.jpg')
        self.cls_result_signal.emit(cls_result)

    # 改
    def label_score(self, file_path):
        print_res = detectItems(file_path)
        # label_score_result = os.path.join(stb_path, clsname + '.jpg')
        self.cls_result_signal.emit(label_score_result)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
