# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow, bg1_abs_path, bg2_abs_path):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1311, 831)
        icon = QtGui.QIcon()
        # 改
        icon.addPixmap(QtGui.QPixmap(bg1_abs_path), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.centralwidget.setObjectName("centralwidget")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(11, 59, 1271, 551))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_4 = QtWidgets.QLabel(self.layoutWidget)
        self.label_4.setAutoFillBackground(True)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout.addWidget(self.label_4)
        self.label_5 = QtWidgets.QLabel(self.layoutWidget)
        self.label_5.setAutoFillBackground(True)
        self.label_5.setStyleSheet("QLabel\n"
"{backkground-color:white\n"
"}")
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout.addWidget(self.label_5)
        self.label_6 = QtWidgets.QLabel(self.layoutWidget)
        self.label_6.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.label_6.setAutoFillBackground(True)
        self.label_6.setStyleSheet("QLabel\n"
"{backkground-color:red\n"
"}")
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout.addWidget(self.label_6)
        self.listView = QtWidgets.QListView(self.centralwidget)
        self.listView.setGeometry(QtCore.QRect(0, 0, 1351, 831))
        # 改
        self.listView.setStyleSheet("border-image: url("+bg2_abs_path+");\n"
"background-position:center;\n"
"background-repeat: no-repeat")
        self.listView.setFlow(QtWidgets.QListView.TopToBottom)
        self.listView.setObjectName("listView")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(390, 650, 200, 40))
        self.pushButton.setStyleSheet("background-color: rgb(0, 0, 255);\n"
"color: rgb(255, 255, 255);\n"
"border-radius: 10px; border: 2px groove gray;\n"
"border-style: outset")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(620, 650, 200, 40))
        self.pushButton_2.setStyleSheet("background-color: rgb(0, 0, 255);\n"
"color: rgb(255, 255, 255);\n"
"border-radius: 10px; border: 2px groove gray;\n"
"border-style: outset")
        self.pushButton_2.setObjectName("pushButton_2")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 30, 365, 20))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(470, 30, 365, 20))
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.listView_2 = QtWidgets.QListView(self.centralwidget)
        self.listView_2.setEnabled(True)
        self.listView_2.setGeometry(QtCore.QRect(1140, 660, 150, 150))
        self.listView_2.setAutoFillBackground(False)
        # 改
        self.listView_2.setStyleSheet("border-image: url("+bg1_abs_path+");\n"
"background-position:center;\n"
"background-repeat: no-repeat;\n"
"")
        self.listView_2.setObjectName("listView_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(620, 710, 200, 40))
        self.pushButton_3.setStyleSheet("background-color: rgb(0, 0, 255);\n"
"color: rgb(255, 255, 255);\n"
"border-radius: 10px; border: 2px groove gray;\n"
"border-style: outset")
        self.pushButton_3.setObjectName("pushButton_3")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(870, 30, 405, 20))
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(390, 710, 200, 40))
        self.pushButton_4.setStyleSheet("background-color: rgb(0, 0, 255);\n"
"color: rgb(255, 255, 255);\n"
"border-radius: 10px; border: 2px groove gray;\n"
"border-style: outset")
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(850, 650, 200, 40))
        self.pushButton_5.setStyleSheet("background-color: rgb(0, 0, 255);\n"
"color: rgb(255, 255, 255);\n"
"border-radius: 10px; border: 2px groove gray;\n"
"border-style: outset")
        self.pushButton_5.setObjectName("pushButton_5")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setEnabled(True)
        self.progressBar.setGeometry(QtCore.QRect(100, 640, 211, 31))
        self.progressBar.setStyleSheet("font: 75 14pt \"Bahnschrift\";")
        self.progressBar.setInputMethodHints(QtCore.Qt.ImhNone)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setTextVisible(True)
        self.progressBar.setObjectName("progressBar")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setEnabled(True)
        self.label_9.setGeometry(QtCore.QRect(50, 770, 1071, 21))
        self.label_9.setAutoFillBackground(True)
        self.label_9.setText("")
        self.label_9.setObjectName("label_9")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(80, 680, 231, 80))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_7 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_7.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.label_7.setAutoFillBackground(False)
        self.label_7.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"color:rgb(255, 0, 0);\n"
"font: 18pt \"Arial\";\n"
"border-style: outset")
        self.label_7.setObjectName("label_7")
        self.verticalLayout.addWidget(self.label_7)
        self.label_8 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_8.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.label_8.setAutoFillBackground(False)
        self.label_8.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"color:rgb(255, 0, 0);\n"
"font: 18pt \"Arial\";\n"
"border-style: outset")
        self.label_8.setObjectName("label_8")
        self.verticalLayout.addWidget(self.label_8)
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(50, 790, 1071, 21))
        self.label_10.setAutoFillBackground(True)
        self.label_10.setText("")
        self.label_10.setObjectName("label_10")
        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_6.setGeometry(QtCore.QRect(850, 710, 101, 40))
        self.pushButton_6.setStyleSheet("background-color: rgb(0, 0, 255);\n"
"color: rgb(255, 255, 255);\n"
"border-radius: 10px; border: 2px groove gray;\n"
"border-style: outset")
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_7 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_7.setGeometry(QtCore.QRect(950, 710, 101, 40))
        self.pushButton_7.setStyleSheet("background-color: rgb(0, 0, 255);\n"
"color: rgb(255, 255, 255);\n"
"border-radius: 10px; border: 2px groove gray;\n"
"border-style: outset")
        self.pushButton_7.setObjectName("pushButton_7")
        self.listView.raise_()
        self.layoutWidget.raise_()
        self.pushButton.raise_()
        self.pushButton_2.raise_()
        self.label.raise_()
        self.label_2.raise_()
        self.listView_2.raise_()
        self.pushButton_3.raise_()
        self.label_3.raise_()
        self.pushButton_4.raise_()
        self.pushButton_5.raise_()
        self.progressBar.raise_()
        self.label_9.raise_()
        self.verticalLayoutWidget.raise_()
        self.label_10.raise_()
        self.pushButton_6.raise_()
        self.pushButton_7.raise_()
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        self.pushButton.clicked.connect(lambda: MainWindow.slc_msg("front"))
        self.pushButton_4.clicked.connect(lambda: MainWindow.slc_msg("back"))

        self.pushButton_2.clicked.connect(MainWindow.cls_msg)
        self.pushButton_3.clicked.connect(MainWindow.label_and_score)

        self.pushButton_5.clicked.connect(MainWindow.reset_all)

        self.pushButton_6.clicked.connect(lambda: MainWindow.show_child("front"))
        self.pushButton_7.clicked.connect(lambda: MainWindow.show_child("back"))

        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "邮票分类与品鉴系统"))
        self.label_4.setText(_translate("MainWindow", "未选择图片"))
        self.label_5.setText(_translate("MainWindow", "未选择图片"))
        self.label_6.setText(_translate("MainWindow", "无分类结果"))
        self.pushButton.setText(_translate("MainWindow", "选择正面"))
        self.pushButton_2.setText(_translate("MainWindow", "开始分类"))
        self.label.setText(_translate("MainWindow", "正面图片"))
        self.label_2.setText(_translate("MainWindow", "背面图片"))
        self.pushButton_3.setText(_translate("MainWindow", "开始品鉴"))
        self.label_3.setText(_translate("MainWindow", "分类结果"))
        self.pushButton_4.setText(_translate("MainWindow", "选择背面"))
        self.pushButton_5.setText(_translate("MainWindow", "重置"))
        self.label_7.setText(_translate("MainWindow", " 打分结果：无"))
        self.label_8.setText(_translate("MainWindow", " 划分等级：无"))
        self.pushButton_6.setText(_translate("MainWindow", "正面标注图"))
        self.pushButton_7.setText(_translate("MainWindow", "背面标注图"))
# import img_rc
# import imgs_rc
