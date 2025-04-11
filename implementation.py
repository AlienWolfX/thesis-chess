import cv2
import torch
from ultralytics import YOLO
import time
from utils.chessFunctions import create_board_display
import numpy as np
from collections import deque
from typing import Dict
import json
from collections import Counter
import csv
from datetime import datetime
import os


device = 'cuda' if torch.cuda.is_available() else 'cpu'

#### SETTINGS ####
model = YOLO('model/v27.pt').to(device)
# Initialize the webcam
# cameraSrc = "https://192.168..207:8080/video" # IP Camera URL
cameraSrc = 0  # Laptop Webcam
cap = cv2.VideoCapture(cameraSrc)
frame_skip = 2 
frame_count = 0

CONFIDENCE_THRESHOLD = 0.40  # Global confidence threshold for detections
BUFFER_SIZE = 10  # Number of frames to keep in buffer
CONSENSUS_THRESHOLD = 0.4  # 40% agreement threshold for stable state detection
USE_CENTER_POINT = False
SHOW_LIVE_WINDOW = True 
MOVES_FILE = 'matches/chess_moves.csv'
last_stable_state = None

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

square_coords = {}

class ChessStateBuffer:
    def __init__(self, buffer_size: int, consensus_threshold: float):
        self.buffer = deque(maxlen=buffer_size)
        self.consensus_threshold = consensus_threshold
        self.current_stable_state = None
    
    def add_state(self, state: Dict[str, str]) -> bool:
        """
        Add a new state to the buffer and check if we have a stable state.
        Returns True if the stable state has changed.
        """
        # Convert dict to frozen set of items for hashability
        state_tuple = frozenset(state.items())
        self.buffer.append(state_tuple)
        
        if len(self.buffer) < self.buffer.maxlen * self.consensus_threshold:
            return False
            
        # Count occurrences of each state
        state_counts = Counter(self.buffer)
        most_common_state = state_counts.most_common(1)[0]
        
        # Check if the most common state appears enough times
        if (most_common_state[1] / len(self.buffer)) >= self.consensus_threshold:
            new_stable_state = dict(most_common_state[0])
            if new_stable_state != self.current_stable_state:
                self.current_stable_state = new_stable_state
                return True
        return False

def get_piece_point(x1: int, y1: int, x2: int, y2: int) -> tuple:
    """Get the reference point for a piece based on global setting."""
    center_x = (x1 + x2) // 2
    if USE_CENTER_POINT:
        center_y = (y1 + y2) // 2  # Use center point
    else:
        center_y = y2  # Use bottom point
    return (center_x, center_y)

def calibrate_board(results):
    piece_positions = {'white_rooks': [], 'black_rooks': []}
    
    for result in results:
        for box in result.boxes:
            if box.conf[0] < CONFIDENCE_THRESHOLD:
                continue
                
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            piece_point = get_piece_point(x1, y1, x2, y2)
            piece_label = model.names[int(box.cls)]
            
            if piece_label == 'White-Rook':
                piece_positions['white_rooks'].append(piece_point)
            elif piece_label == 'Black-Rook':
                piece_positions['black_rooks'].append(piece_point)
    
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
        inner_spacing = 7.7  # Increase for tighter inner squares
        
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
            if box.conf[0] < CONFIDENCE_THRESHOLD:
                continue
                
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            piece_point = get_piece_point(x1, y1, x2, y2)
            piece_label = model.names[int(box.cls)]
            
            # Find closest square using piece point
            closest_square = None
            min_distance = float('inf')
            for square, coord in square_coords.items():
                distance = np.sqrt((piece_point[0] - coord[0])**2 + 
                                 (piece_point[1] - coord[1])**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_square = square
            
            if closest_square and min_distance < 50: 
                current_positions[closest_square] = class_id_mapping[piece_label]
                print(f'{piece_label} detected at {closest_square}')
    
    return current_positions

def save_move_to_csv(old_state: Dict[str, str], new_state: Dict[str, str]) -> None:
    """
    Compare two board states and save the detected move to CSV.
    """
    if old_state is None:
        return
        
    # Find differences between states
    moved_from = None
    moved_to = None
    piece_moved = None
    
    for square in set(old_state.keys()) | set(new_state.keys()):
        old_piece = old_state.get(square)
        new_piece = new_state.get(square)
        
        if old_piece != new_piece:
            if old_piece and not new_piece:
                moved_from = square
                piece_moved = old_piece
            elif new_piece and not old_piece:
                moved_to = square
    
    if moved_from and moved_to and piece_moved:
        # Determine if it's a white or black move
        player = "White" if piece_moved.isupper() else "Black"
        move = f"{moved_from}-{moved_to}"
        
        # Create CSV file if it doesn't exist
        file_exists = os.path.isfile(MOVES_FILE)
        with open(MOVES_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Timestamp', 'Player', 'Move'])
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                           player, 
                           move])

# Main loop
state_buffer = ChessStateBuffer(BUFFER_SIZE, CONSENSUS_THRESHOLD)
calibrated = False

# Declare global variable before using it
last_stable_state = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    resized_frame = cv2.resize(frame, (620, 540))
    results = model(resized_frame)
    
    # Only create annotated frame if we're showing live window
    annotated_frame = resized_frame.copy() if SHOW_LIVE_WINDOW else None

    if not calibrated:
        calibrated = calibrate_board(results)
    
    if SHOW_LIVE_WINDOW:
        draw_grid(annotated_frame, square_coords)
    
    if calibrated:
        current_positions = track_pieces(results)
        
        # Add state to buffer and check for stable state
        if state_buffer.add_state(current_positions):
            stable_state = state_buffer.current_stable_state
            print("New stable board state detected:")
            print(json.dumps(stable_state, indent=2))
            
            # Save move to CSV when board state changes
            if last_stable_state is not None:
                save_move_to_csv(last_stable_state, stable_state)
            last_stable_state = stable_state.copy()
            
        if state_buffer.current_stable_state:
            board_vis = create_board_display(state_buffer.current_stable_state)
        else:
            board_vis = create_board_display(current_positions)
            
        # Position the board window
        cv2.namedWindow('Chess Board', cv2.WINDOW_NORMAL)
        cv2.moveWindow('Chess Board', 200, 200)
        cv2.imshow('Chess Board', board_vis)
        
        # Only process visualization if live window is enabled
        if SHOW_LIVE_WINDOW:
            for result in results:
                for box in result.boxes:
                    if box.conf[0] < CONFIDENCE_THRESHOLD: 
                        continue
                        
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    piece_label = model.names[int(box.cls)]
                    piece_point = get_piece_point(x1, y1, x2, y2)
                    
                    # Draw bounding box and label
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
                    cv2.putText(annotated_frame, piece_label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                    
                    # Find closest square
                    closest_square = None
                    min_distance = float('inf')
                    for square, coord in square_coords.items():
                        distance = np.sqrt((piece_point[0] - coord[0])**2 + 
                                         (piece_point[1] - coord[1])**2)
                        if distance < min_distance:
                            min_distance = distance
                            closest_square = square
                    
                    if closest_square and min_distance < 50:
                        # Draw connection line to mapped square
                        mapped_pos = square_coords[closest_square]
                        cv2.line(annotated_frame, piece_point, mapped_pos, (0, 255, 255), 1)
                        cv2.circle(annotated_frame, mapped_pos, 4, (0, 255, 255), -1)

    if SHOW_LIVE_WINDOW:
        draw_grid(annotated_frame, square_coords)
        cv2.putText(annotated_frame, "Calibrated" if calibrated else "Calibrating...", 
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('Live', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()