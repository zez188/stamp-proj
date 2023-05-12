# -*- coding: UTF-8 -*-

import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget
from PyQt5.Qt import QPixmap, QPoint, Qt, QPainter
dict_side = {"front": "正面", "back": "背面"}
class ImageBox(QWidget):
    def __init__(self):
        super(ImageBox, self).__init__()
        self.img = None
        self.scaled_img = None
        self.point = QPoint(0, 0)
        self.start_pos = None
        self.end_pos = None
        self.left_click = False
        self.scale = 1

    # def init_ui(self):
    #     self.setWindowTitle("ImageBox")

    def set_image(self, img_path):
        """
        open image file
        :param img_path: image file path
        :return:
        """
        # img = QImageReader(img_path)
        # img.setScaledSize(QSize(self.size().width(), self.size().height()))
        # img = img.read()
        self.img = QPixmap(img_path).scaled(3000, 3000, Qt.KeepAspectRatio)
        self.scaled_img = self.img
        self.update()

    def paintEvent(self, e):
        """
        receive paint events
        :param e: QPaintEvent
        :return:
        """
        if self.scaled_img:
            painter = QPainter()
            painter.begin(self)
            painter.scale(self.scale, self.scale)
            painter.drawPixmap(self.point, self.scaled_img)
            painter.end()

    def wheelEvent(self, event):
        angle = event.angleDelta() / 8  # 返回QPoint对象，为滚轮转过的数值，单位为1/8度
        angleY = angle.y()
        # 获取当前鼠标相对于view的位置
        if angleY > 0:
            self.scale *= 1.1
        else:  # 滚轮下滚
            self.scale *= 0.9
        self.adjustSize()
        self.update()

    def mouseMoveEvent(self, e):
        """
        mouse move events for the widget
        :param e: QMouseEvent
        :return:
        """
        if self.left_click:
            self.end_pos = 1.25*(e.pos() - self.start_pos)
            self.point = self.point + self.end_pos
            self.start_pos = e.pos()
            self.repaint()

    def mousePressEvent(self, e):
        """
        mouse press events for the widget
        :param e: QMouseEvent
        :return:
        """
        if e.button() == Qt.LeftButton:
            self.left_click = True
            self.start_pos = e.pos()

    def mouseReleaseEvent(self, e):
        """
        mouse release events for the widget
        :param e: QMouseEvent
        :return:
        """
        if e.button() == Qt.LeftButton:
            self.left_click = False

class Child(ImageBox):

    def __init__(self, labeled_img_path, score_list, side):
        super().__init__()
        self.setWindowTitle("未加扣,"+dict_side[side]+"："+str(score_list))
        self.init_ui(labeled_img_path)
        self.center()

        self.scrollArea = None
        self.scrollAreaWidgetContents = None
        self.gridLayout1 = None
        self.box = None
        self.gridLayout = None

        # self.setWindowState(QtCore.Qt.WindowMaximized)

    def init_ui(self, labeled_img_path):
        self.scrollArea = QtWidgets.QScrollArea()
        self.scrollArea.setGeometry(QtCore.QRect(240, 50, 1200, 800))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        # 定义一个总布局
        self.gridLayout1 = QtWidgets.QVBoxLayout()
        self.scrollAreaWidgetContents.setLayout(self.gridLayout1)

        # self.box是绘图类
        self.box = ImageBox()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, self.box.width(), self.box.height()))
        self.scrollAreaWidgetContents.setMinimumSize(QtCore.QSize(1200, 800))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")

        # 子布局
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        # 将绘图类添加进子布局
        self.gridLayout.addWidget(self.box, 0, 0, 1, 1)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.open_image(labeled_img_path)

        # 将子布局加入总布局
        self.gridLayout1.addLayout(self.gridLayout)
        self.setMinimumSize(1200, 800)
        # 将总布局设置为当前布局文件
        self.setLayout(self.gridLayout1)

    def open_image(self, image_path):
        """
        select image file and open it
        :return:
        """
        img = QPixmap(image_path)
        # print(img.width(),"+",img.height())
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, img.width(), img.height()))
        self.box.set_image(img)

    # 屏幕居中
    def center(self):
        # 获取窗口左上角位置
        # left_top_pos_x = self.x()
        # left_top_pos_y = self.y()
        # 获取窗口大小
        screen = QtWidgets.QDesktopWidget().screenGeometry()
        print(screen.width(), screen.height())
        size = self.geometry()
        print(size.width(), size.height())
        # 本窗体运动
        self.move((screen.width() - size.width() - 2*self.x()) / 2, (screen.height() - size.height() - 2*self.y()) / 2)





