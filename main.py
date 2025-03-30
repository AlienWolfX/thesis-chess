"""
The main file for the project.
"""

from PyQt6.QtWidgets import QApplication, QMainWindow, QDialog, QFileDialog, QListWidgetItem
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtGui import QPixmap, QImage, QPainter
from PyQt6.QtCore import Qt
from forms.ui_mainWindow import Ui_MainWindow
from forms.ui_newGame import Ui_newGame
from forms.ui_viewHistory import Ui_viewHistoryx
from forms.ui_about import Ui_about
import cv2 as cv
import chess
import chess.svg
import os
import pandas as pd

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.newGame.clicked.connect(self.newGame)
        self.ui.viewHistory.clicked.connect(self.viewHistory)
        self.ui.menuHelp.triggered.connect(self.menuHelp)

    def newGame(self):
        self.newGameDialog = NewGame(self)
        self.newGameDialog.exec()
        
    def viewHistory(self):
        self.viewHistoryWindow = ViewHistory(self)
        self.viewHistoryWindow.show()
        
    def menuHelp(self):
        self.aboutDialog = About(self)
        self.aboutDialog.exec()

class NewGame(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_newGame()
        self.ui.setupUi(self)
        self.ui.okButton.clicked.connect(self.close)

class ViewHistory(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_viewHistoryx()
        self.ui.setupUi(self)
        self.chessboard = chess.Board()
        self.board_size = 600 
        
        self.ui.chessOutput.setMinimumSize(self.board_size, self.board_size)
        
        # Connect signals
        self.ui.loadButton.clicked.connect(self.loadMatch)
        self.ui.matchList.itemClicked.connect(self.displayMatch)
        self.ui.movesList.itemClicked.connect(self.displayMove)
        self.ui.movesList.currentRowChanged.connect(self.displayMove)
        # Add new button connections
        self.ui.resetButton.clicked.connect(self.resetView)
        self.ui.returnButton.clicked.connect(self.close)
        
        # Run Methods
        self.displayMatches()
        self.renderChessboard(initial=True)

    def resetView(self):
        """Reset the view to initial state"""
        # Clear move list
        self.ui.movesList.clear()
        
        # Clear match selection
        self.ui.matchList.clearSelection()
        
        # Reset chessboard to starting position
        self.chessboard.reset()
        
        # Render initial board
        self.renderChessboard(initial=True)
        
        # Clear any stored moves
        if hasattr(self, 'moves_df'):
            del self.moves_df
        
        # Reset current move index
        self.current_move_index = 0

    # Display csv files on matches folder
    def displayMatches(self):
        matches_folder = os.path.join(os.getcwd(), "matches")
        if os.path.exists(matches_folder):
            for file_name in os.listdir(matches_folder):
                if file_name.endswith(".csv"):
                    self.ui.matchList.addItem(file_name)

    def displayMatch(self, item):
        match_file = item.text()
        matches_folder = os.path.join(os.getcwd(), "matches")
        match_path = os.path.join(matches_folder, match_file)
        if os.path.exists(match_path):
            self.fetch_moves(match_path)
            self.display_moves_list()

    # Open file if wala sa matches folder naka store ang csv
    def loadMatch(self):
        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Open File",
            os.path.expanduser("~"),
            "CSV files (*.csv);;All Files (*)"
        )
        if fname:
            file_name = os.path.basename(fname)
            self.ui.matchList.addItem(file_name)
                
    def fetch_moves(self, csv_path):
        self.moves_df = pd.read_csv(csv_path)
        self.chessboard.reset()
        self.current_move_index = 0

    def display_moves_list(self):
        self.ui.movesList.clear()
        for index, row in self.moves_df.iterrows():
            move_item = QListWidgetItem(f"{row['timestamp']}: {row['move']}")
            move_item.setData(Qt.ItemDataRole.UserRole, index)
            self.ui.movesList.addItem(move_item)
    
    def displayMove(self, index):
        if isinstance(index, QListWidgetItem):
            move_index = index.data(Qt.ItemDataRole.UserRole)
        else:
            move_index = index

        self.chessboard.reset()
        for i in range(move_index + 1):
            move = self.moves_df.iloc[i]['move']
            self.chessboard.push_san(move)
        self.renderChessboard()

    def renderChessboard(self, initial=False):
        # Create board with fixed size for both initial and move displays
        board_size = {'size': self.board_size}
        
        if initial:
            chessboard_svg = chess.svg.board(self.chessboard, **board_size).encode("UTF-8")
        else:
            last_move = self.chessboard.peek() if self.chessboard.move_stack else None
            arrows = []
            if last_move:
                arrows.append(chess.svg.Arrow(last_move.from_square, last_move.to_square, color='red'))
            chessboard_svg = chess.svg.board(self.chessboard, arrows=arrows, **board_size).encode("UTF-8")

        # Create renderer with fixed size
        svg_renderer = QSvgRenderer(chessboard_svg)
        
        # Create image with board size
        image = QImage(self.board_size, self.board_size, QImage.Format.Format_ARGB32)
        image.fill(0x00ffffff)
        painter = QPainter(image)
        svg_renderer.render(painter)
        painter.end()
        
        # Set pixmap directly at board size
        pixmap = QPixmap.fromImage(image)
        self.ui.chessOutput.setPixmap(pixmap)
        
        # Ensure dialog size accommodates the board
        self.adjustSize()

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