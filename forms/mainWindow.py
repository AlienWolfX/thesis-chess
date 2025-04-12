# Form implementation generated from reading ui file 'ui/mainWindow.ui'
#
# Created by: PyQt6 UI code generator 6.8.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(928, 600)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("ui\\../img/app.ico"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.appName = QtWidgets.QLabel(parent=self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semibold")
        font.setPointSize(36)
        font.setBold(True)
        font.setWeight(75)
        self.appName.setFont(font)
        self.appName.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.appName.setObjectName("appName")
        self.verticalLayout_2.addWidget(self.appName)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.newGame = QtWidgets.QPushButton(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.newGame.sizePolicy().hasHeightForWidth())
        self.newGame.setSizePolicy(sizePolicy)
        self.newGame.setMinimumSize(QtCore.QSize(0, 24))
        self.newGame.setObjectName("newGame")
        self.verticalLayout.addWidget(self.newGame)
        self.viewHistory = QtWidgets.QPushButton(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.viewHistory.sizePolicy().hasHeightForWidth())
        self.viewHistory.setSizePolicy(sizePolicy)
        self.viewHistory.setObjectName("viewHistory")
        self.verticalLayout.addWidget(self.viewHistory)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 928, 21))
        self.menubar.setObjectName("menubar")
        self.menuMenu = QtWidgets.QMenu(parent=self.menubar)
        self.menuMenu.setObjectName("menuMenu")
        self.menuHelp = QtWidgets.QMenu(parent=self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionReport_Issue = QtGui.QAction(parent=MainWindow)
        self.actionReport_Issue.setObjectName("actionReport_Issue")
        self.actionAbout = QtGui.QAction(parent=MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.actionNew_Game = QtGui.QAction(parent=MainWindow)
        self.actionNew_Game.setObjectName("actionNew_Game")
        self.actionExport_to_CSV = QtGui.QAction(parent=MainWindow)
        self.actionExport_to_CSV.setObjectName("actionExport_to_CSV")
        self.actionOpen_CSV = QtGui.QAction(parent=MainWindow)
        self.actionOpen_CSV.setObjectName("actionOpen_CSV")
        self.menuMenu.addAction(self.actionNew_Game)
        self.menuMenu.addSeparator()
        self.menuMenu.addAction(self.actionOpen_CSV)
        self.menuHelp.addAction(self.actionAbout)
        self.menubar.addAction(self.menuMenu.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "CheckMate"))
        self.appName.setText(_translate("MainWindow", "RookEye.ph"))
        self.newGame.setText(_translate("MainWindow", "New Game"))
        self.viewHistory.setText(_translate("MainWindow", "View History"))
        self.menuMenu.setTitle(_translate("MainWindow", "File"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.actionReport_Issue.setText(_translate("MainWindow", "Report Issue"))
        self.actionAbout.setText(_translate("MainWindow", "About"))
        self.actionNew_Game.setText(_translate("MainWindow", "New Game"))
        self.actionExport_to_CSV.setText(_translate("MainWindow", "Export to CSV"))
        self.actionOpen_CSV.setText(_translate("MainWindow", "Open CSV"))
