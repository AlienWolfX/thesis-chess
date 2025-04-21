from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtGui import QPixmap, QImage, QPainter
import chess
import chess.svg
import chess.pgn
import pandas as pd
import os

def render_chessboard(board, board_size, initial=False):
    """Render chess board as QPixmap"""
    board_options = {'size': board_size}
    
    if initial:
        chessboard_svg = chess.svg.board(board, **board_options).encode("UTF-8")
    else:
        last_move = board.peek() if board.move_stack else None
        arrows = []
        if last_move:
            arrows.append(chess.svg.Arrow(last_move.from_square, last_move.to_square, color='red'))
        chessboard_svg = chess.svg.board(board, arrows=arrows, **board_options).encode("UTF-8")

    svg_renderer = QSvgRenderer(chessboard_svg)
    image = QImage(board_size, board_size, QImage.Format.Format_ARGB32)
    image.fill(0x00ffffff)
    painter = QPainter(image)
    svg_renderer.render(painter)
    painter.end()
    
    return QPixmap.fromImage(image)

def load_pgn_moves(pgn_path):
    """Load and process moves from PGN file"""
    with open(pgn_path) as pgn_file:
        game = chess.pgn.read_game(pgn_file)
        if game:
            moves_data = []
            node = game
            move_number = 1
            
            while node.next():
                node = node.next()
                san_move = node.san()
                is_white = len(moves_data) % 2 == 0
                moves_data.append({
                    'Move': san_move,
                    'Player': 'White' if is_white else 'Black',
                    'MoveNumber': f"{move_number}{'.' if is_white else ''}"
                })
                if not is_white:
                    move_number += 1
            
            return pd.DataFrame(moves_data), game.headers
    return None, None

def load_csv_moves(csv_path):
    """Load and process moves from CSV file"""
    moves_df = pd.read_csv(csv_path)
    headers = {
        'Event': os.path.splitext(os.path.basename(csv_path))[0],
        'Date': moves_df.iloc[0]['Timestamp'].split()[0] if len(moves_df) > 0 else '',
        'White': 'White',
        'Black': 'Black'
    }
    return moves_df, headers