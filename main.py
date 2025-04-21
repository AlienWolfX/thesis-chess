"""
GUI for the project.
"""

from PyQt6.QtWidgets import QApplication, QMainWindow, QDialog, QFileDialog, QListWidgetItem, QMessageBox
from PyQt6.QtCore import Qt
from forms.mainWindow import Ui_MainWindow
from forms.newGame import Ui_newGame
from forms.viewHistory import Ui_viewHistory
from forms.chessGame import Ui_ChessGameWindow
from forms.about import Ui_about
import cv2 as cv
import chess
import os
import pandas as pd
from utils.view_history_utils import render_chessboard, load_pgn_moves, load_csv_moves
from utils.validation_utils import validate_alphanumeric, validate_alphanumeric_with_spaces, validate_numeric

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
                print(f"Round: {game_details['round']}")
                print(f"Site: {game_details['site']}")
                
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
        self.ui = Ui_ChessGameWindow()
        self.ui.setupUi(self)
        
        # Set game information
        self.ui.gameNameLabel.setText(f"Game: {game_details['game_name']}")
        self.ui.roundLabel.setText(f"Round: {game_details['round']}")
        self.ui.siteLabel.setText(f"Site: {game_details['site']}")
        
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
            'black_player': '',
            'round': '',
            'site': '' 
        }
        
        # Populate camera combo box
        self.populate_cameras()
        
        # Connect buttons and validation
        self.ui.proceedButton.clicked.connect(self.validateAndSave)
        self.ui.gameNameEdit.textChanged.connect(self.validateGameName)
        self.ui.whitePlayerEdit.textChanged.connect(self.validatePlayerName)
        self.ui.blackPlayerEdit.textChanged.connect(self.validatePlayerName)
        self.ui.roundEdit.textChanged.connect(self.validateRound)
        self.ui.siteEdit.textChanged.connect(self.validateSite)

    def populate_cameras(self):
        """Detect and populate available cameras"""
        self.ui.cameraComboBox.clear()
        
        # Test cameras from index 0 to 99
        for i in range(100):
            cap = cv.VideoCapture(i, cv.CAP_DSHOW)  # Use DirectShow on Windows
            if cap.isOpened():
                # Get camera info
                ret, frame = cap.read()
                if ret:
                    # Get camera resolution
                    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
                    camera_text = f"Camera {i} ({width}x{height})"
                else:
                    camera_text = f"Camera {i}"
                    
                self.ui.cameraComboBox.addItem(camera_text, i)
                cap.release()
            else:
                # No more cameras found, break the loop
                break
        
        # Select first camera by default if available
        if self.ui.cameraComboBox.count() > 0:
            self.ui.cameraComboBox.setCurrentIndex(0)

    def validateGameName(self):
        """Validate game name to allow alphanumeric characters"""
        game_name = self.ui.gameNameEdit.text()
        if not validate_alphanumeric(game_name):
            self.ui.gameNameEdit.setText(game_name[:-1])

    def validatePlayerName(self):
        """Validate player names to allow alphanumeric characters and spaces"""
        for edit in [self.ui.whitePlayerEdit, self.ui.blackPlayerEdit]:
            player_name = edit.text()
            if not validate_alphanumeric_with_spaces(player_name):
                edit.setText(player_name[:-1])

    def validateRound(self):
        """Validate round to only allow integers"""
        round_text = self.ui.roundEdit.text()
        if not validate_numeric(round_text):
            self.ui.roundEdit.setText(round_text[:-1])

    def validateSite(self):
        """Validate site to allow alphanumeric characters and spaces"""
        site = self.ui.siteEdit.text()
        if not validate_alphanumeric_with_spaces(site):
            self.ui.siteEdit.setText(site[:-1])

    def validateAndSave(self):
        """Validate all fields before saving"""
        game_name = self.ui.gameNameEdit.text()
        white_player = self.ui.whitePlayerEdit.text()
        black_player = self.ui.blackPlayerEdit.text()
        round_number = self.ui.roundEdit.text()
        site = self.ui.siteEdit.text()

        # Check if fields are empty
        if not all([game_name, white_player, black_player, round_number, site]):  # Include site
            QMessageBox.warning(self, "Validation Error", "All fields must be filled")
            return

        # Additional validation
        if not game_name.isalnum():
            QMessageBox.warning(self, "Validation Error", "Game name must contain only letters and numbers")
            return

        if not all(c.isalnum() or c.isspace() for c in white_player + black_player):
            QMessageBox.warning(self, "Validation Error", "Player names must contain only letters, numbers and spaces")
            return

        if not round_number.isdigit():
            QMessageBox.warning(self, "Validation Error", "Round must be a number")
            return

        # Add site validation
        if not all(c.isalnum() or c.isspace() for c in site):
            QMessageBox.warning(self, "Validation Error", "Site must contain only letters, numbers and spaces")
            return

        # If all validations pass, save and accept
        self.saveGameDetails()

    def saveGameDetails(self):
        """Save game details before closing"""
        self.game_details['game_name'] = self.ui.gameNameEdit.text()
        self.game_details['white_player'] = self.ui.whitePlayerEdit.text()
        self.game_details['black_player'] = self.ui.blackPlayerEdit.text()
        self.game_details['round'] = self.ui.roundEdit.text()
        self.game_details['site'] = self.ui.siteEdit.text() 
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
        
        # Clear Game information labels
        self.ui.gameNoLabel.setText("")
        self.ui.labelDate.setText("")
        self.ui.whitePlayerLabel.setText("")
        self.ui.blackPlayerLabel.setText("")
        
        # Reset chessboard to starting position
        self.chessboard.reset()
        
        # Render initial board
        self.renderChessboard(initial=True)
        
        # Clear any stored moves
        if hasattr(self, 'moves_df'):
            del self.moves_df
        
        # Reset current move index
        self.current_move_index = 0

    def displayMatches(self):
        """Display only PGN files from matches folder"""
        matches_folder = os.path.join(os.getcwd(), "matches")
        if os.path.exists(matches_folder):
            for file_name in os.listdir(matches_folder):
                if file_name.endswith(".pgn"):  # Only look for PGN files
                    item = QListWidgetItem(file_name)
                    self.ui.matchList.addItem(item)

    def loadMatch(self):
        """Load and display moves from a PGN file"""
        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Open File",
            os.path.expanduser("~"),
            "Chess Games (*.pgn)"  # Only allow PGN files
        )
        if fname:
            matches_folder = os.path.join(os.getcwd(), "matches")
            if not os.path.exists(matches_folder):
                os.makedirs(matches_folder)
                
            file_name = os.path.basename(fname)
            match_path = os.path.join(matches_folder, file_name)
            
            if fname != match_path:
                import shutil
                shutil.copy2(fname, match_path)
            
            item = QListWidgetItem(file_name)
            self.ui.matchList.addItem(item)
            self.displayMatch(item)

    def displayMatch(self, item):
        """Display moves from selected PGN match"""
        if not item:
            return
            
        match_file = item.text()
        matches_folder = os.path.join(os.getcwd(), "matches")
        match_path = os.path.join(matches_folder, match_file)
        
        if os.path.exists(match_path):
            try:
                self.fetch_moves_from_pgn(match_path)
                self.display_moves_list()
                
                if self.ui.movesList.count() > 0:
                    self.ui.movesList.setCurrentRow(0)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load moves: {str(e)}")

    def fetch_moves_from_pgn(self, pgn_path):
        """Read and process moves from PGN file"""
        moves_df, headers = load_pgn_moves(pgn_path)
        if moves_df is not None:
            self.moves_df = moves_df
            
            # Update game information labels
            self.ui.gameNoLabel.setText(headers.get('Event', ''))
            self.ui.labelDate.setText(f"Date: {headers.get('Date', '')}")
            self.ui.whitePlayerLabel.setText(headers.get('White', 'White'))
            self.ui.blackPlayerLabel.setText(headers.get('Black', 'Black'))
            
            self.chessboard.reset()
            self.current_move_index = 0

    def fetch_moves_from_csv(self, csv_path):
        """Read and process moves from CSV file"""
        self.moves_df, headers = load_csv_moves(csv_path)
        
        # Update game information labels
        self.ui.gameNoLabel.setText(headers.get('Event', ''))
        self.ui.labelDate.setText(f"Date: {headers.get('Date', '')}")
        self.ui.whitePlayerLabel.setText(headers.get('White', 'White'))
        self.ui.blackPlayerLabel.setText(headers.get('Black', 'Black'))
        
        self.chessboard.reset()
        self.current_move_index = 0

    def display_moves_list(self):
        """Display moves in the moves list"""
        self.ui.movesList.clear()
        for index, row in self.moves_df.iterrows():
            # Format the move display based on file type
            if 'MoveNumber' in row:  # PGN format
                move_text = f"{row['MoveNumber']} {row['Move']}"
            else:  # CSV format
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
            try:
                # Handle PGN format moves (already in SAN notation)
                self.chessboard.push_san(move)
            except ValueError as e:
                try:
                    # Fallback for CSV format ('e2-e4' style)
                    from_square, to_square = move.split('-')
                    chess_move = chess.Move.from_uci(from_square + to_square)
                    if chess_move in self.chessboard.legal_moves:
                        self.chessboard.push(chess_move)
                    else:
                        QMessageBox.warning(self, "Error", f"Illegal move detected: {move}")
                        break
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Invalid move format: {move}")
                    break
                    
        self.renderChessboard()

    def renderChessboard(self, initial=False):
        pixmap = render_chessboard(self.chessboard, self.board_size, initial)
        self.ui.chessOutput.setPixmap(pixmap)
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