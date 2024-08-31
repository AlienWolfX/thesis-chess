"""
The main file for the project.
"""

from PyQt6.QtWidgets import QApplication, QMainWindow, QDialog
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import pyqtSignal, QThread, Qt
from ui_mainWindow import Ui_MainWindow
from ui_newGame import Ui_newGame
from ui_viewHistory import Ui_viewHistory
from ui_about import Ui_about
from ui_calibration import Ui_calibrateCamera
import cv2 as cv

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.newGame.clicked.connect(self.newGame)
        self.ui.viewHistory.clicked.connect(self.viewHistory)
        self.ui.menuAbout.triggered.connect(self.menuAbout)

    def newGame(self):
        self.newGameDialog = NewGame(self)
        self.newGameDialog.exec()
        
    def viewHistory(self):
        self.viewHistoryWindow = ViewHistory(self)
        self.viewHistoryWindow.show()
        
    def menuAbout(self):
        self.aboutDialog = About(self)
        self.aboutDialog.exec()

class NewGame(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_newGame()
        self.ui.setupUi(self)
        self.ui.okButton.clicked.connect(self.close)
        self.ui.calibrateCameraButton.clicked.connect(self.calibrateCamera)
        
    def calibrateCamera(self):
        self.calibrationDialog = Calibration(self)
        self.calibrationDialog.exec()

class Calibration(QDialog):
    ImageUpdate = pyqtSignal(QImage)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_calibrateCamera()
        self.ui.setupUi(self)

        self.cap = cv.VideoCapture(0)
        self.ThreadActive = False

        self.ImageUpdate.connect(self.imageUpdate)

        self.startCamera()

    def startCamera(self):
        self.ThreadActive = True
        self.worker = CameraWorker(self.cap)
        self.worker.ImageUpdate.connect(self.imageUpdate)
        self.worker.start()

    def stopCamera(self):
        self.ThreadActive = False
        self.worker.stop()
        self.cap.release() 
        
    def imageUpdate(self, image):
        self.ui.imageOutput.setPixmap(QPixmap.fromImage(image))

    def closeEvent(self, event):
        self.stopCamera() 
        super().closeEvent(event)

class CameraWorker(QThread):
    ImageUpdate = pyqtSignal(QImage)

    def __init__(self, cap):
        super().__init__()
        self.cap = cap
        self.ThreadActive = True

    def run(self):
        while self.ThreadActive:
            ret, frame = self.cap.read()
            if ret:
                flip = cv.flip(frame, 1)
                rgbImage = cv.cvtColor(flip, cv.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
                self.ImageUpdate.emit(p)

    def stop(self):
        self.ThreadActive = False
        self.quit()
        self.wait()

class ViewHistory(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_viewHistory()
        self.ui.setupUi(self)

class About(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_about()
        self.ui.setupUi(self)
        self.ui.okButton.clicked.connect(self.close)

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()