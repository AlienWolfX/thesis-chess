"""
The main file for the project.
"""

from PyQt6.QtWidgets import QApplication, QMainWindow, QDialog, QFileDialog, QListWidgetItem, QMessageBox
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtGui import QPixmap, QImage, QPainter
from PyQt6.QtCore import Qt
from forms.mainWindow import Ui_MainWindow
from forms.newGame import Ui_newGame
from forms.viewHistory import Ui_viewHistory
from forms.about import Ui_about
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
        if self.newGameDialog.exec() == QDialog.DialogCode.Accepted:
            game_details = self.newGameDialog.getGameDetails()
            if game_details:
                print(f"Starting new game: {game_details['game_name']}")
                print(f"White player: {game_details['white_player']}")
                print(f"Black player: {game_details['black_player']}")
                # Add your game initialization code here
        
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
        
        # Initialize game details
        self.game_details = {
            'game_name': '',
            'white_player': '',
            'black_player': ''
        }
        
        # Connect buttons and validation
        self.ui.proceedButton.clicked.connect(self.validateAndSave)
        self.ui.gameNameEdit.textChanged.connect(self.validateGameName)
        self.ui.whitePlayerEdit.textChanged.connect(self.validatePlayerName)
        self.ui.blackPlayerEdit.textChanged.connect(self.validatePlayerName)

    def validateGameName(self):
        """Validate game name to only allow numbers"""
        game_name = self.ui.gameNameEdit.text()
        if not game_name.isdigit() and game_name != '':
            self.ui.gameNameEdit.setText(game_name[:-1])

    def validatePlayerName(self):
        """Validate player names to only allow letters and spaces"""
        for edit in [self.ui.whitePlayerEdit, self.ui.blackPlayerEdit]:
            player_name = edit.text()
            if not all(c.isalpha() or c.isspace() for c in player_name) and player_name != '':
                edit.setText(player_name[:-1])

    def validateAndSave(self):
        """Validate all fields before saving"""
        game_name = self.ui.gameNameEdit.text()
        white_player = self.ui.whitePlayerEdit.text()
        black_player = self.ui.blackPlayerEdit.text()

        # Check if fields are empty
        if not all([game_name, white_player, black_player]):
            QMessageBox.warning(self, "Validation Error", "All fields must be filled")
            return

        # Additional validation
        if not game_name.isdigit():
            QMessageBox.warning(self, "Validation Error", "Game name must contain only numbers")
            return

        if not all(c.isalpha() or c.isspace() for c in white_player + black_player):
            QMessageBox.warning(self, "Validation Error", "Player names must contain only letters")
            return

        # If all validations pass, save and accept
        self.saveGameDetails()

    def saveGameDetails(self):
        """Save game details before closing"""
        self.game_details['game_name'] = self.ui.gameNameEdit.text()
        self.game_details['white_player'] = self.ui.whitePlayerEdit.text()
        self.game_details['black_player'] = self.ui.blackPlayerEdit.text()
        self.accept()

    def getGameDetails(self):
        """Return the game details if dialog was accepted"""
        return self.game_details if self.result() == QDialog.DialogCode.Accepted else None

class ViewHistory(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_viewHistory()
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
        
        # Sa updating game information planning to change this based sa csv file
        self.ui.gameNoLabel.setText("5")
        self.ui.whitePlayerLabel.setText("Magnus Carlsen")
        self.ui.blackPlayerLabel.setText("Hikaru Nakamura")
        self.ui.labelDate.setText("Date: 2023-11-20")
        
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