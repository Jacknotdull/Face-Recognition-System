# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UserProfile.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_UserProfile(object):
    def setupUi(self, UserProfile):
        UserProfile.setObjectName("UserProfile")
        UserProfile.resize(1001, 609)
        self.cam = QtWidgets.QLabel(UserProfile)
        self.cam.setGeometry(QtCore.QRect(20, 10, 960, 540))
        self.cam.setStyleSheet("background-color: rgb(186, 186, 186);")
        self.cam.setText("")
        self.cam.setObjectName("cam")
        self.nameinput = QtWidgets.QLineEdit(UserProfile)
        self.nameinput.setGeometry(QtCore.QRect(260, 560, 221, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.nameinput.setFont(font)
        self.nameinput.setObjectName("nameinput")
        self.label_2 = QtWidgets.QLabel(UserProfile)
        self.label_2.setGeometry(QtCore.QRect(210, 560, 47, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.confirm = QtWidgets.QPushButton(UserProfile)
        self.confirm.setGeometry(QtCore.QRect(610, 560, 91, 31))
        self.confirm.setObjectName("confirm")
        self.returnmenu = QtWidgets.QPushButton(UserProfile)
        self.returnmenu.setGeometry(QtCore.QRect(890, 560, 91, 31))
        self.returnmenu.setObjectName("returnmenu")
        self.capture = QtWidgets.QPushButton(UserProfile)
        self.capture.setGeometry(QtCore.QRect(510, 560, 91, 31))
        self.capture.setObjectName("capture")

        self.retranslateUi(UserProfile)
        QtCore.QMetaObject.connectSlotsByName(UserProfile)

    def retranslateUi(self, UserProfile):
        _translate = QtCore.QCoreApplication.translate
        UserProfile.setWindowTitle(_translate("UserProfile", "User Profile"))
        self.label_2.setText(_translate("UserProfile", "Name:"))
        self.confirm.setText(_translate("UserProfile", "Save"))
        self.returnmenu.setText(_translate("UserProfile", "Return"))
        self.capture.setText(_translate("UserProfile", "Take photo"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    UserProfile = QtWidgets.QWidget()
    ui = Ui_UserProfile()
    ui.setupUi(UserProfile)
    UserProfile.show()
    sys.exit(app.exec_())
