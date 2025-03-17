import chess
import chess.svg
import cv2
import numpy as np
from PIL import Image
import io
import cairosvg

def create_fen_from_positions(current_positions):
    """Convert current piece positions to FEN string"""
    board = [['' for _ in range(8)] for _ in range(8)]
    
    # Fill known piece positions
    for square, piece in current_positions.items():
        file = ord(square[0]) - ord('a')
        rank = 8 - int(square[1])
        if 0 <= rank < 8 and 0 <= file < 8:
            board[rank][file] = piece
    
    # Convert to FEN
    fen = []
    for rank in board:
        empty = 0
        rank_str = ''
        for square in rank:
            if square == '':
                empty += 1
            else:
                if empty > 0:
                    rank_str += str(empty)
                    empty = 0
                rank_str += square
        if empty > 0:
            rank_str += str(empty)
        fen.append(rank_str)
    
    return '/'.join(fen)

def fen_to_image(fen, size=400):
    """Convert FEN string to chess board image"""
    try:
        # Create chess board from FEN
        board = chess.Board(fen)
        
        # Generate SVG
        svg_content = chess.svg.board(
            board=board,
            size=size,
            coordinates=True,
            colors={
                'square light': '#FFFFFF',
                'square dark': '#86A666',
                'margin': '#242424',
                'coord': '#FFFFFF',
            }
        )
        
        # Convert SVG to PNG using cairosvg
        png_data = cairosvg.svg2png(bytestring=svg_content.encode('utf-8'))
        
        # Convert to OpenCV format
        image = cv2.imdecode(
            np.frombuffer(png_data, np.uint8),
            cv2.IMREAD_COLOR
        )
        
        return image
    
    except Exception as e:
        print(f"Error creating board visualization: {e}")
        # Return a blank image on error
        return np.zeros((size, size, 3), dtype=np.uint8)

def create_board_display(current_positions, size=600):
    """Create board display from current positions"""
    # Convert positions to FEN
    fen = create_fen_from_positions(current_positions)
    if not fen:
        fen = '8/8/8/8/8/8/8/8'  # Empty board
    
    # Generate board image
    return fen_to_image(fen, size)