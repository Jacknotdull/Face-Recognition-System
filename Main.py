from PyQt5 import QtCore, QtGui, QtWidgets
from LoginPage import Ui_LoginPage
from Menu import Ui_Menu
from RecogGUI import Ui_Rcognition
from Log import Ui_Log
from UserProfile import Ui_UserProfile
import os
import cv2
import numpy as np
import face_recognition
from datetime import datetime
import smtplib
from email.message import EmailMessage

class LoginWindow(QtWidgets.QMainWindow, Ui_LoginPage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self.loginbutton.clicked.connect(self.authentication)

    def authentication(self):
        adminUsername = os.environ.get("Admin_username")
        adminPassword = os.environ.get("Admin_password")
        if self.username.text() == adminUsername and self.password.text() == adminPassword:
            msg = QtWidgets.QMessageBox()
            msg.information(self, 'Welcome', 'Login successful')
            msg.buttonClicked.connect(self.prompt)
            msg.exec_()
            LoginPage.hide()
        else:
            QtWidgets.QMessageBox.critical(self, 'Try again', 'Incorrect username or password')

    def prompt(self):
        self.MenuPage = MenuWindow()
        self.MenuPage.show()

class MenuWindow(QtWidgets.QWidget, Ui_Menu):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self.Recognition.clicked.connect(self.recognition)
        self.Viewlog.clicked.connect(self.viewLog)
        self.Newuser.clicked.connect(self.newUser)
        self.Logout.clicked.connect(self.closeEvent)

    def recognition(self):
        self.RecognitionPage = RecWindow()
        self.RecognitionPage.show()
        self.RecognitionPage.startRec()

    def viewLog(self):
        self.LogPage = ViewWindow()
        self.LogPage.show()

    def newUser(self):
        self.UserPage = UserWindow()
        self.UserPage.show()
        self.UserPage.startCap()

    def closeEvent(self):
        quit_msg = "Are you sure you want to log out?"
        reply = QtWidgets.QMessageBox.question(self, 'Message', quit_msg, QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            app.exit()
        else:
            pass

class RecWindow(QtWidgets.QWidget, Ui_Rcognition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self.stopbutton.clicked.connect(self.stopRec)

    def stopRec(self):
        self.timer.stop()
        self.cap.release()
        self.hide()

    def startRec(self):
        self.cap = cv2.VideoCapture(0) #starting point
        self.timer = QtCore.QTimer(self)
        path = 'Images'
        images = []  # store images
        self.knownList = []
        self.classNames = []  # store names
        myList = os.listdir(path)  # get the name list
        for m in myList:
            curr = cv2.imread(f'{path}/{m}')  # read image
            images.append(curr)  # store image
            self.classNames.append(os.path.splitext(m)[0])  # store name without extension
        for i in images:
            i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB) # convert color to rgb
            encodeface = face_recognition.face_encodings(i)[0] # encode the face found
            self.knownList.append(encodeface) # store the encoding
        self.timer.timeout.connect(self.updateFrame)
        self.timer.start(10)

    def updateFrame(self):
        ret, image = self.cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:
            faceLoc = face_recognition.face_locations(image)
            encodedface = face_recognition.face_encodings(image, faceLoc)
            # encode the face found in current frame

            for enc, fl in zip(encodedface, faceLoc):
                match = face_recognition.compare_faces(self.knownList, enc, tolerance=0.54) # compare the face by comparing the encoding
                dis = face_recognition.face_distance(self.knownList, enc)  # return false probability
                top, right, bottom, left = fl # coordinate of the face
                print(match)
                print(dis)
                index = np.argmin(dis)  # get the index of lowest distance
                if True in match:
                    if match[index]:
                        name = self.classNames[index]
                        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 255, 0))
                        cv2.putText(image, name.upper(), (left + 6, bottom - 6),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

                    with open('EntryLog.csv', 'r+') as f:
                        # read and write file with pointer at the beginning of the file
                        myList = f.readlines()
                        mydict = {}
                        for line in myList:
                            entry = line.split(',')
                            mydict[entry[0]] = entry[1] + ' ' + entry[2].strip()
                        if name not in mydict: # if the name not in csv file
                            now = datetime.now()
                            date = now.strftime('%Y-%m-%d')
                            time = now.strftime('%H:%M:%S')
                            f.writelines(f'\n{name},{date},{time}')
                        elif name in mydict: # if name exists then compare the last access time
                            x = datetime.strptime(mydict[name], '%Y-%m-%d %H:%M:%S')
                            now = datetime.now()
                            difference = now - x
                            if difference.total_seconds() > 59:
                                date = now.strftime('%Y-%m-%d')
                                time = now.strftime('%H:%M:%S')
                                f.writelines(f'\n{name},{date},{time}')
                else:
                    name = 'unknown' # if detected unknown face then label it with
                    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 255, 0))
                    cv2.putText(image, name.upper(), (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                (255, 255, 255), 1)
                    imm = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    imm = imm[top:top+bottom, left:left+right]
                    self.mailTrigger(imm) # send unknown face image to trigger email
        except Exception as e:
            print(e)
        height, width, channel = image.shape
        bytes_per_line = channel * width
        image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.imagelabel.setPixmap(QtGui.QPixmap.fromImage(image)) # display image to the interface
        self.imagelabel.setScaledContents(True)

    def mailTrigger(self, image):
        id = len(os.listdir("Unknown"))+1
        cv2.imwrite(f'Unknown\\unknown{id}.jpg', image)
        admail = os.environ.get('Admin_mail')
        adpass = os.environ.get('Admin_mailpass')
        msg = EmailMessage()
        msg['Subject'] = 'Unknown Person Alert!'
        msg['From'] = admail
        msg['To'] = 'TP051511@mail.apu.edu.my'
        msg.set_content('Unknown face image attached.')
        with open(f'Unknown\\unknown{id}.jpg', 'rb') as f:  # read byte
            file_data = f.read()
        msg.add_attachment(file_data, maintype='image', subtype='jpg', filename='unknown face')
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:  # context manager to manage
            smtp.login(admail, adpass)
            smtp.send_message(msg)

class UserWindow(QtWidgets.QWidget, Ui_UserProfile):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self.logic = 0
        self.capture.clicked.connect(self.captureImage)
        self.nameinput.textChanged.connect(self.nameChecker)
        self.confirm.clicked.connect(self.saveNew)
        self.confirm.setEnabled(False)
        self.returnmenu.clicked.connect(self.returnMenu)

    def startCap(self):
        self.cap = cv2.VideoCapture(0)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.updateFrame)
        self.timer.start(40)

    def updateFrame(self):
        ret, image = self.cap.read()
        if self.logic == 1:
            return image
        elif self.logic == 0:
            pass
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width, channel = image.shape
        bytes_per_line = channel * width
        image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.cam.setPixmap(QtGui.QPixmap.fromImage(image))
        self.cam.setScaledContents(True)

    def captureImage(self):
        self.logic = 1
        self.timer.stop()

    def saveNew(self):
        reply = QtWidgets.QMessageBox.question(self, 'Message', "Do you want to save this user?",
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            try:
                image = self.updateFrame()
                name = self.nameinput.text()
                cv2.imwrite(f"Images\\{name}.jpg", image)
                msg = QtWidgets.QMessageBox()
                ret = msg.information(self, 'Profile saved!',
                                      f'New user {name.upper()} added successfully. Do you want to add another user?',
                                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.Yes)
                if ret == QtWidgets.QMessageBox.Yes:
                    self.logic = 0
                    self.nameinput.clear()
                    self.timer.timeout.connect(self.updateFrame)
                    self.timer.start(40)
                else:
                    self.returnMenu()
            except Exception as e:
                self.msg.warning(self, 'Profile failed to save', "Please try again")
        else:
            self.logic = 0
            self.nameinput.clear()
            self.timer.timeout.connect(self.updateFrame)
            self.timer.start(40)

    def nameChecker(self):
        self.confirm.setEnabled(True)

    def returnMenu(self):
        reply = QtWidgets.QMessageBox.question(self, 'Message', "Are you sure you want to leave this page?",
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            self.timer.stop()
            self.cap.release()
            self.close()
        else:
            pass

class ViewWindow(QtWidgets.QWidget, Ui_Log):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self.loadData()

    def loadData(self):
        with open('EntryLog.csv', 'r+') as f:
            lines = f.readlines()
            f.close()
            self.tableWidget.setRowCount(len(lines))
            for i in range(0, len(lines)):
                tokens = lines[i].strip().split(",")
                self.tableWidget.setItem(i, 0, QtWidgets.QTableWidgetItem(tokens[0]))
                self.tableWidget.setItem(i, 1, QtWidgets.QTableWidgetItem(tokens[1]))
                self.tableWidget.setItem(i, 2, QtWidgets.QTableWidgetItem(tokens[2]))

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    LoginPage = LoginWindow()
    LoginPage.show()
    app.exec_()