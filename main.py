"""
The main file for the project.
"""

from PyQt6.QtWidgets import QApplication, QMainWindow, QDialog
from ui_mainWindow import Ui_MainWindow
from ui_newGame import Ui_newGame
from ui_viewHistory import Ui_viewHistory

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.newGame.clicked.connect(self.newGame)
        self.ui.viewHistory.clicked.connect(self.viewHistory)

    def newGame(self):
        self.newGameDialog = NewGame(self)
        self.newGameDialog.exec()
        
    def viewHistory(self):
        self.viewHistoryWindow = ViewHistory(self)
        self.viewHistoryWindow.show()

class NewGame(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_newGame()
        self.ui.setupUi(self)

class ViewHistory(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_viewHistory()
        self.ui.setupUi(self)

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()