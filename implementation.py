import cv2
import torch
from ultralytics import YOLO
import time
from utils.chessFunctions import fen_to_image, create_board_display
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = YOLO('model/v20.pt').to(device)

# Initialize the webcam or IP camera
# cameraSrc = "https://192.168.6.207:8080/video" # IP Camera URL
cameraSrc = 1  # Laptop Webcam

cap = cv2.VideoCapture(cameraSrc)

frame_skip = 2 
frame_count = 0

# Define a mapping of class names to FEN labels
class_id_mapping = {
    'Black-Bishop': 'b',
    'Black-King': 'k',
    'Black-Knight': 'n',
    'Black-Pawn': 'p',
    'Black-Queen': 'q',
    'Black-Rook': 'r',
    'White-Bishop': 'B',
    'White-King': 'K',
    'White-Knight': 'N',
    'White-Pawn': 'P',
    'White-Queen': 'Q',
    'White-Rook': 'R',
}

initial_board = {
    'a1': 'R', 'b1': 'N', 'c1': 'B', 'd1': 'Q', 'e1': 'K', 'f1': 'B', 'g1': 'N', 'h1': 'R',
    'a2': 'P', 'b2': 'P', 'c2': 'P', 'd2': 'P', 'e2': 'P', 'f2': 'P', 'g2': 'P', 'h2': 'P',
    'a7': 'p', 'b7': 'p', 'c7': 'p', 'd7': 'p', 'e7': 'p', 'f7': 'p', 'g7': 'p', 'h7': 'p',
    'a8': 'r', 'b8': 'n', 'c8': 'b', 'd8': 'q', 'e8': 'k', 'f8': 'b', 'g8': 'n', 'h8': 'r'
}

# Display the starting board
start = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'
board = fen_to_image(start)
# board_image = cv2.imread('current_board.png')

square_coords = {}

def calibrate_board(results):
    piece_positions = {'white_rooks': [], 'black_rooks': []}
    
    # Detect all rooks
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            center_x = (x1 + x2) // 2
            center_y = y2  # Always use bottom of bounding box
            piece_label = model.names[int(box.cls)]
            
            piece_data = (center_x, center_y)
            
            if piece_label == 'White-Rook':
                piece_positions['white_rooks'].append(piece_data)
            elif piece_label == 'Black-Rook':
                piece_positions['black_rooks'].append(piece_data)
    
    # Need all 4 rooks for calibration
    if len(piece_positions['white_rooks']) == 2 and len(piece_positions['black_rooks']) == 2:
        # Sort rooks by x-coordinate
        white_rooks = sorted(piece_positions['white_rooks'], key=lambda x: x[0])
        black_rooks = sorted(piece_positions['black_rooks'], key=lambda x: x[0])
        
        # Assign corners based on position (bottom to top)
        h1 = white_rooks[0]  # Bottom right
        a1 = white_rooks[1]  # Bottom left
        a8 = black_rooks[0]  # Top left
        h8 = black_rooks[1]  # Top right
        
        square_coords['a1'] = a1
        square_coords['h1'] = h1
        square_coords['a8'] = a8
        square_coords['h8'] = h8
        
        # Calculate rank positions with adjusted interpolation
        outer_spacing = 7.0  # Keep outer bounds
        inner_spacing = 7.0  # Increase for tighter inner squares
        
        for rank in range(1, 9):
            # Use outer spacing for overall board dimensions
            rank_ratio = (rank - 1) / outer_spacing
            
            # Get left and right points for this rank
            left_x = int(a1[0] + rank_ratio * (a8[0] - a1[0]))
            left_y = int(a1[1] + rank_ratio * (a8[1] - a1[1]))
            right_x = int(h1[0] + rank_ratio * (h8[0] - h1[0]))
            right_y = int(h1[1] + rank_ratio * (h8[1] - h1[1]))
            
            # Interpolate points along this rank with tighter inner spacing
            for file_idx, file in enumerate('abcdefgh'):
                # Adjust file ratio for tighter inner squares
                file_ratio = file_idx / inner_spacing
                x = int(left_x + file_ratio * (right_x - left_x))
                y = int(left_y + file_ratio * (right_y - left_y))
                square_coords[f'{file}{rank}'] = (x, y)
        
        return True
    return False

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

def draw_grid(frame, square_coords):
    if all(k in square_coords for k in ['a1', 'h1', 'a8', 'h8']):
        a1 = square_coords['a1']
        h1 = square_coords['h1']
        a8 = square_coords['a8']
        h8 = square_coords['h8']

        # Draw outer boundary
        cv2.line(frame, a1, a8, (0, 255, 0), 1)  # Left vertical
        cv2.line(frame, h1, h8, (0, 255, 0), 1)  # Right vertical
        cv2.line(frame, a1, h1, (0, 255, 0), 1)  # Bottom horizontal
        cv2.line(frame, a8, h8, (0, 255, 0), 1)  # Top horizontal

        # Draw internal vertical lines
        for file_idx, file in enumerate('bcdefg'):
            ratio = (file_idx + 1) / 7.0
            
            # Get bottom point
            bottom_x = int(a1[0] + ratio * (h1[0] - a1[0]))
            bottom_y = int(a1[1] + ratio * (h1[1] - a1[1]))
            
            # Get top point
            top_x = int(a8[0] + ratio * (h8[0] - a8[0]))
            top_y = int(a8[1] + ratio * (h8[1] - a8[1]))
            
            # Draw vertical line
            cv2.line(frame, (bottom_x, bottom_y), (top_x, top_y), (0, 255, 0), 1)

        # Draw internal horizontal lines
        for rank in range(1, 7):
            ratio = rank / 7.0
            
            # Get left point
            left_x = int(a1[0] + ratio * (a8[0] - a1[0]))
            left_y = int(a1[1] + ratio * (a8[1] - a1[1]))
            
            # Get right point
            right_x = int(h1[0] + ratio * (h8[0] - h1[0]))
            right_y = int(h1[1] + ratio * (h8[1] - h1[1]))
            
            # Draw horizontal line
            cv2.line(frame, (left_x, left_y), (right_x, right_y), (0, 255, 0), 1)

        # Draw intersection points and labels
        for file_idx, file in enumerate('abcdefgh'):
            for rank in range(1, 9):
                square = f'{file}{rank}'
                pos = square_coords[square]
                
                # Draw intersection point
                cv2.circle(frame, pos, 2, (0, 255, 0), -1)
                
                # Draw square label
                cv2.putText(frame, square, (pos[0]-10, pos[1]+15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

        # Mark corner points
        for pos, label in [('a1', 'A1'), ('h1', 'H1'), ('a8', 'A8'), ('h8', 'H8')]:
            point = square_coords[pos]
            cv2.circle(frame, point, 4, (0, 0, 255), -1)
            cv2.putText(frame, label, (point[0]-10, point[1]+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
def track_pieces(results):
    current_positions = {}
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            center_x = (x1 + x2) // 2
            center_y = y2 
            piece_label = model.names[int(box.cls)]
            
            # Find closest square using bottom center of bounding box
            closest_square = None
            min_distance = float('inf')
            for square, coord in square_coords.items():
                distance = np.sqrt((center_x - coord[0])**2 + (center_y - coord[1])**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_square = square
            
            if closest_square and min_distance < 50: 
                current_positions[closest_square] = class_id_mapping[piece_label]
                print(f'{piece_label} detected at {closest_square}')
    
    return current_positions

def detect_moves(previous_positions, current_positions):
    """Detect chess piece movements by comparing positions"""
    moves = []
    
    # Find pieces that moved (either new positions or disappeared)
    for square, piece in current_positions.items():
        if square not in previous_positions or previous_positions[square] != piece:
            for old_square, old_piece in previous_positions.items():
                if old_piece == piece and old_square != square:
                    moves.append((piece, old_square, square))
                    break
    
    return moves

previous_positions = {}

# Main loop
calibrated = False
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    resized_frame = cv2.resize(frame, (620, 540))
    results = model(resized_frame)
    annotated_frame = resized_frame.copy()

    if not calibrated:
        calibrated = calibrate_board(results)
    
    draw_grid(annotated_frame, square_coords)
    
    if calibrated:
        current_positions = track_pieces(results)
        
        # Create and show board visualization
        board_vis = create_board_display(current_positions)
        
        # Position the board window
        cv2.namedWindow('Chess Board', cv2.WINDOW_NORMAL)
        cv2.moveWindow('Chess Board', 200, 200)
        cv2.imshow('Chess Board', board_vis)
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                piece_label = model.names[int(box.cls)]
                center_x = (x1 + x2) // 2
                center_y = y2
                
                # Draw bounding box and label
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                cv2.putText(annotated_frame, piece_label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                
                # Find closest square
                closest_square = None
                min_distance = float('inf')
                for square, coord in square_coords.items():
                    distance = np.sqrt((center_x - coord[0])**2 + (center_y - coord[1])**2)
                    if distance < min_distance:
                        min_distance = distance
                        closest_square = square
                
                if closest_square and min_distance < 50:
                    # Draw connection line to mapped square
                    mapped_pos = square_coords[closest_square]
                    cv2.line(annotated_frame, (center_x, center_y), mapped_pos, (0, 255, 255), 1)
                    cv2.circle(annotated_frame, mapped_pos, 4, (0, 255, 255), -1)
                    
        # Detect and display moves
        if previous_positions:
            move_text = []
            for square, piece in current_positions.items():
                if square not in previous_positions or previous_positions[square] != piece:
                    # Find source square
                    for prev_square, prev_piece in previous_positions.items():
                        if prev_piece == piece and prev_square != square:
                            # Draw move arrow
                            start = square_coords[prev_square]
                            end = square_coords[square]
                            cv2.arrowedLine(annotated_frame, start, end, (0, 0, 255), 2)
                            
                            # Add move text
                            move_text.append(f"{piece}: {prev_square}->{square}")
            
            # Display moves in window
            for idx, text in enumerate(move_text):
                cv2.putText(annotated_frame, text, (10, 50 + 20*idx),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        previous_positions = current_positions.copy()

    draw_grid(annotated_frame, square_coords)
    cv2.putText(annotated_frame, "Calibrated" if calibrated else "Calibrating...", 
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Live', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()