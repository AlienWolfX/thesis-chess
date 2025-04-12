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
                
                # Open the ChessGameWindow and pass the game details
                self.chessGameWindow = ChessGameWindow(game_details)
                self.chessGameWindow.show()
        
    def viewHistory(self):
        self.viewHistoryWindow = ViewHistory(self)
        self.viewHistoryWindow.show()
        
    def menuHelp(self):
        self.aboutDialog = About(self)
        self.aboutDialog.exec()

class ChessGameWindow(QMainWindow):
    def __init__(self, game_details, parent=None):
        super().__init__(parent)
        from forms.chessgame import Ui_ChessGameWindow
        self.ui = Ui_ChessGameWindow()
        self.ui.setupUi(self)
        
        # Set the game details in the UI
        self.ui.player1Name.setText(game_details['white_player'])
        self.ui.player2Name.setText(game_details['black_player'])
        
        # Connect the End Game button to close the window
        self.ui.endButton.clicked.connect(self.endGame)

    def endGame(self):
        """Close the ChessGameWindow and return to MainWindow."""
        self.close()

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
        self.ui.resetButton.clicked.connect(self.resetView)
        self.ui.returnButton.clicked.connect(self.close)
        
        # Initialize game state
        self.current_move_index = 0
        self.moves = []
        
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

    def loadMatch(self):
        """Load and display moves from a CSV file"""
        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Open File",
            os.path.expanduser("~"),
            "CSV files (*.csv);;All Files (*)"
        )
        if fname:
            # Copy file to matches folder if it doesn't exist there
            matches_folder = os.path.join(os.getcwd(), "matches")
            if not os.path.exists(matches_folder):
                os.makedirs(matches_folder)
                
            file_name = os.path.basename(fname)
            match_path = os.path.join(matches_folder, file_name)
            
            # Copy file if it's not already in matches folder
            if fname != match_path:
                import shutil
                shutil.copy2(fname, match_path)
            
            # Add to list and display moves
            item = QListWidgetItem(file_name)
            self.ui.matchList.addItem(item)
            self.displayMatch(item)

    def displayMatch(self, item):
        """Display moves from selected match"""
        if not item:
            return
            
        match_file = item.text()
        matches_folder = os.path.join(os.getcwd(), "matches")
        match_path = os.path.join(matches_folder, match_file)
        
        if os.path.exists(match_path):
            try:
                # Load and display moves
                self.fetch_moves(match_path)
                self.display_moves_list()
                
                # Select first move
                if self.ui.movesList.count() > 0:
                    self.ui.movesList.setCurrentRow(0)
                    
                print(f"Loaded {self.ui.movesList.count()} moves from {match_file}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load moves: {str(e)}")

    def fetch_moves(self, csv_path):
        """Read and process moves from CSV file"""
        self.moves_df = pd.read_csv(csv_path)
        
        # Update game information
        if len(self.moves_df) > 0:
            first_move = self.moves_df.iloc[0]
            last_move = self.moves_df.iloc[-1]
            
            # Update game information labels
            self.ui.gameNoLabel.setText(os.path.splitext(os.path.basename(csv_path))[0])
            self.ui.labelDate.setText(f"Date: {first_move['Timestamp'].split()[0]}")
            
            # Track players based on their moves
            white_moves = self.moves_df[self.moves_df['Player'] == 'White']
            black_moves = self.moves_df[self.moves_df['Player'] == 'Black']
            
            if not white_moves.empty:
                self.ui.whitePlayerLabel.setText("White")
            if not black_moves.empty:
                self.ui.blackPlayerLabel.setText("Black")
        
        self.chessboard.reset()
        self.current_move_index = 0

    def display_moves_list(self):
        """Display moves in the moves list"""
        self.ui.movesList.clear()
        for index, row in self.moves_df.iterrows():
            # Format the move display
            timestamp = row['Timestamp'].split()[1]  # Get only the time part
            move = row['Move']
            player = row['Player']
            move_text = f"{index + 1}. {player}: {move} ({timestamp})"
            
            move_item = QListWidgetItem(move_text)
            move_item.setData(Qt.ItemDataRole.UserRole, index)
            self.ui.movesList.addItem(move_item)

    def displayMove(self, index):
        """Display the selected move on the chessboard"""
        if not hasattr(self, 'moves_df') or self.moves_df.empty:
            return
            
        if isinstance(index, QListWidgetItem):
            move_index = index.data(Qt.ItemDataRole.UserRole)
        else:
            move_index = index

        if move_index < 0:
            return

        self.chessboard.reset()
        for i in range(move_index + 1):
            move = self.moves_df.iloc[i]['Move']
            # Convert the move format from 'e2-e4' to 'e2e4'
            from_square, to_square = move.split('-')
            chess_move = chess.Move.from_uci(from_square + to_square)
            if chess_move in self.chessboard.legal_moves:
                self.chessboard.push(chess_move)
            else:
                QMessageBox.warning(self, "Error", f"Illegal move detected: {move}")
                break
                
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