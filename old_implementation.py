import cv2
import torch
from ultralytics import YOLO
from utils.chessFunctions import create_board_display
import numpy as np
from collections import deque
from typing import Dict
import json
from collections import Counter
import csv
from datetime import datetime
import os
import chess
import chess.pgn
import chess.svg
import cairosvg
import io
from PIL import Image
import psutil
import threading
import time
from time import perf_counter
from threading import Lock

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#### SETTINGS ####
model = YOLO('model/v29.pt').to(device)
cameraSrc = 0 
cap = cv2.VideoCapture(cameraSrc)
frame_skip = 3
frame_count = 0
WIDTH = 800
HEIGHT = 800

CONFIDENCE_THRESHOLD = 0.40
BUFFER_SIZE = 8
CONSENSUS_THRESHOLD = 0.4
USE_CENTER_POINT = False
SHOW_LIVE_WINDOW = True 
ENABLE_RESOURCE_MONITORING = False
gameName = datetime.now().strftime("%Y%m%d_%H%M")
MOVES_FILE = f'matches/game_{gameName}_.csv'
last_stable_state = None

calibrated = False
square_coords = {}

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

def ensure_directories():
    """Ensure required directories exist"""
    os.makedirs("matches", exist_ok=True)

class PGNRecorder:
    def __init__(self, game_name: str, white_player: str = "White", black_player: str = "Black"):
        self.game = chess.pgn.Game()
        self.game.headers["Event"] = f"Game {game_name}"
        self.game.headers["Site"] = "Chess Tracking System"
        self.game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        self.game.headers["Round"] = "1"
        self.game.headers["White"] = white_player
        self.game.headers["Black"] = black_player
        self.game.headers["Result"] = "*"
        
        self.board = chess.Board()
        self.node = self.game
        self.moves_played = 0
        self.current_move_number = 1
        self.is_white_move = True
        self.move_history = [] 

    def format_move_number(self) -> str:
        """Return the current move number if it's white's turn"""
        return f"{self.current_move_number}." if self.is_white_move else ""
    
    def add_move_to_history(self, san_move: str) -> None:
        """Add formatted move to history"""
        if self.is_white_move:
            self.move_history.append(f"{self.current_move_number}. {san_move}")
        else:
            self.move_history.append(san_move)

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

class ResourceMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.running = False
        self.process = psutil.Process()
        self.log_file = f'resource_usage_{datetime.now().strftime("%Y%m%d_%H%M")}.csv'
        self.fps_buffer = deque(maxlen=30)
        self.inference_times = deque(maxlen=30)
        self.start_time = perf_counter()
        
        # Create/open CSV file with expanded headers
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp', 
                'CPU Usage (%)', 
                'Memory Usage (MB)',
                'FPS',
                'Inference Time (ms)',
                'Uptime (s)'
            ])
    
    def add_performance_metrics(self, frame_time: float, inference_time: float):
        """Add frame and inference times to buffers"""
        self.fps_buffer.append(frame_time)
        self.inference_times.append(inference_time)
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()
    
    def _monitor(self):
        while self.running:
            try:
                # Get CPU and memory usage
                cpu_percent = self.process.cpu_percent()
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                
                # Calculate FPS and average inference time
                current_fps = 1.0 / (sum(self.fps_buffer) / len(self.fps_buffer)) if self.fps_buffer else 0
                avg_inference = (sum(self.inference_times) / len(self.inference_times) * 1000) if self.inference_times else 0
                uptime = perf_counter() - self.start_time
                
                # Write to CSV
                with open(self.log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        f"{cpu_percent:.1f}",
                        f"{memory_mb:.1f}",
                        f"{current_fps:.1f}",
                        f"{avg_inference:.1f}",
                        f"{int(uptime)}"
                    ])
                
                time.sleep(self.interval)
            except Exception as e:
                print(f"Error monitoring resources: {e}")
                break

class ChessDisplay:
    def __init__(self, size=800):
        self.size = size
        self.current_state = None
        self.cached_image = None
        self.display_lock = Lock()
        
    def update(self, board_state):
        """Thread-safe update with double buffering"""
        if board_state == self.current_state:
            return self.cached_image
            
        with self.display_lock:
            # Create new image in background
            board = chess.Board()
            board.clear()
            
            for square, piece in board_state.items():
                square_idx = chess.parse_square(square)
                piece_obj = chess.Piece.from_symbol(piece)
                board.set_piece_at(square_idx, piece_obj)
            
            svg_content = chess.svg.board(board=board, size=self.size)
            png_data = cairosvg.svg2png(bytestring=svg_content.encode('utf-8'))
            image = Image.open(io.BytesIO(png_data))
            new_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGR)
            
            # Swap buffers
            self.cached_image = new_image
            self.current_state = board_state.copy() if board_state else None
            
        return self.cached_image

def get_piece_point(x1: int, y1: int, x2: int, y2: int, is_rook_calibration: bool = False) -> tuple:
    """
    Get the reference point for a piece.
    Args:
        x1, y1: Top-left corner of bounding box
        x2, y2: Bottom-right corner of bounding box
        is_rook_calibration: If True, use bottom point for calibration rooks
    Returns:
        tuple: (x, y) coordinates of reference point
    """
    center_x = (x1 + x2) // 2
    
    if is_rook_calibration:
        # Use bottom point for calibration rooks
        return (center_x, y2)
    else:
        # Use weighted point between center and bottom for normal tracking
        center_y = (y1 + y2) // 2
        bottom_y = y2
        piece_y = (center_y + bottom_y) // 2
        return (center_x, piece_y)

def calibrate_board(results):
    """Calibrate the chess board using rook bottom positions"""
    piece_positions = {'white_rooks': [], 'black_rooks': []}
    
    for result in results:
        for box in result.boxes:
            if box.conf[0] < CONFIDENCE_THRESHOLD:
                continue
                
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            piece_label = model.names[int(box.cls)]
            
            if piece_label == 'White-Rook':
                # Use bottom point for calibration rooks
                piece_point = get_piece_point(x1, y1, x2, y2, is_rook_calibration=True)
                piece_positions['white_rooks'].append(piece_point)
            elif piece_label == 'Black-Rook':
                piece_point = get_piece_point(x1, y1, x2, y2, is_rook_calibration=True)
                piece_positions['black_rooks'].append(piece_point)
    
    # Need all 4 rooks for calibration
    if len(piece_positions['white_rooks']) == 2 and len(piece_positions['black_rooks']) == 2:
        print("Found all 4 rooks, calibrating grid...")
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
    else:
        print(f"Waiting for all rooks: White={len(piece_positions['white_rooks'])}/2, Black={len(piece_positions['black_rooks'])}/2")
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
                
class_confidence_thresholds = {
    'Black-King': 0.7  # Higher confidence threshold for Black King
}

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
            
            # Vectorized distance calculation
            coords = np.array(list(square_coords.values()))
            distances = np.sqrt(np.sum((coords - np.array(piece_point))**2, axis=1))
            min_distance = np.min(distances)
            if min_distance < 50:
                closest_square = list(square_coords.keys())[np.argmin(distances)]
                current_positions[closest_square] = class_id_mapping[piece_label]
    
    return current_positions

def is_reachable_state(old_state: Dict[str, str], new_state: Dict[str, str], max_moves: int = 2) -> bool:
    """
    Check if new_state is reachable from old_state within max_moves moves.
    Returns True if reachable, False otherwise.
    """
    if old_state is None:
        return True  # First state is always valid
        
    # Create board from old state
    old_board = chess.Board()
    old_board.clear()
    for square, piece in old_state.items():
        square_idx = chess.parse_square(square)
        piece_obj = chess.Piece.from_symbol(piece)
        old_board.set_piece_at(square_idx, piece_obj)
    
    # Create board from new state
    new_board = chess.Board()
    new_board.clear()
    for square, piece in new_state.items():
        square_idx = chess.parse_square(square)
        piece_obj = chess.Piece.from_symbol(piece)
        new_board.set_piece_at(square_idx, piece_obj)
    
    # Count differences between boards
    differences = 0
    for square in chess.SQUARES:
        old_piece = old_board.piece_at(square)
        new_piece = new_board.piece_at(square)
        if old_piece != new_piece:
            differences += 1
    
    # Each move changes at most 2 squares (from and to)
    # So max_moves * 2 is the maximum number of squares that can change
    return differences <= (max_moves * 2)

def save_move_to_csv_and_pgn(old_state: Dict[str, str], new_state: Dict[str, str], pgn_recorder: PGNRecorder) -> None:
    """Compare two board states and store the detected move."""
    if old_state is None:
        return
        
    # Check if the new state is reachable
    if not is_reachable_state(old_state, new_state):
        print("Warning: New board state is not reachable within 2 moves - ignoring")
        return
        
    # Find differences between states
    moved_from = None
    moved_to = None
    piece_moved = None
    captured_piece = None
    
    for square in set(old_state.keys()) | set(new_state.keys()):
        old_piece = old_state.get(square)
        new_piece = new_state.get(square)
        
        if old_piece != new_piece:
            if old_piece and not new_piece:
                moved_from = square
                piece_moved = old_piece
            elif new_piece:
                moved_to = square
                # If there was a piece in the destination square, it was captured
                if square in old_state:
                    captured_piece = old_state[square]
    
    if moved_from and moved_to and piece_moved:
        try:
            # Create chess move (python-chess will handle capture notation)
            chess_move = chess.Move.from_uci(f"{moved_from}{moved_to}")
            if chess_move in pgn_recorder.board.legal_moves:
                # Get algebraic notation (will include 'x' for captures)
                san_move = pgn_recorder.board.san(chess_move)
                
                # Add move to history
                pgn_recorder.add_move_to_history(san_move)
                
                # Update game state
                pgn_recorder.node = pgn_recorder.node.add_variation(chess_move)
                pgn_recorder.board.push(chess_move)
                pgn_recorder.moves_played += 1
                
                # Update move counter
                if not piece_moved.isupper():  # After black's move
                    pgn_recorder.current_move_number += 1
                pgn_recorder.is_white_move = not pgn_recorder.is_white_move
                
                move_text = san_move
                if captured_piece:
                    print(f"Capture detected: {piece_moved} takes {captured_piece} on {moved_to}")
                print(f"Move {move_text} recorded. Total moves: {pgn_recorder.moves_played}")
                
        except Exception as e:
            print(f"Error recording move: {e}")

def finalize_game(pgn_recorder: PGNRecorder, result: str = "*") -> None:
    """Finalize the game and save both PGN and CSV files."""
    try:
        # Update game result
        pgn_recorder.game.headers["Result"] = result
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        game_name = pgn_recorder.game.headers["Event"].replace("Game ", "")
        
        # Save PGN file
        pgn_filename = f"matches/game_{game_name}_final.pgn"
        with open(pgn_filename, "w") as f:
            print(pgn_recorder.game, file=f)
            
        # Save CSV file with move history
        ensure_directories()
        with open(MOVES_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Game', 'Timestamp', 'Moves'])
            for move in pgn_recorder.move_history:
                writer.writerow([
                    game_name,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    move
                ])
            
        print(f"Game saved to {pgn_filename}")
        print(f"Total moves recorded: {pgn_recorder.moves_played}")
        
    except Exception as e:
        print(f"Error saving files: {e}")

def reset_calibration():
    """Reset calibration state and related variables"""
    global calibrated, square_coords, last_stable_state
    calibrated = False
    square_coords.clear()
    last_stable_state = None
    return ChessStateBuffer(BUFFER_SIZE, CONSENSUS_THRESHOLD)

# Main loop
state_buffer = ChessStateBuffer(BUFFER_SIZE, CONSENSUS_THRESHOLD)

# Declare global variable before using it
last_stable_state = None

# Initialize PGN recorder before the main loop
ensure_directories()
pgn_recorder = PGNRecorder(
    game_name=datetime.now().strftime("%Y%m%d_%H%M"),
    white_player="White",
    black_player="Black"
)

# Initialize resource monitor if enabled
resource_monitor = None
if ENABLE_RESOURCE_MONITORING:
    resource_monitor = ResourceMonitor(interval=1.0)
    resource_monitor.start()

# Initialize ChessDisplay
chess_display = ChessDisplay(size=800)

while True:
    frame_start = perf_counter()
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    resized_frame = cv2.resize(frame, (WIDTH, HEIGHT), 
                             interpolation=cv2.INTER_AREA)
    
    # Measure inference time
    inference_start = perf_counter()
    results = model(resized_frame)
    inference_time = perf_counter() - inference_start
    
    # Calculate frame time and update resource monitor
    frame_time = perf_counter() - frame_start
    if resource_monitor:
        resource_monitor.add_performance_metrics(frame_time, inference_time)
    
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
            
            # Save move to CSV and PGN when board state changes
            if last_stable_state is not None:
                save_move_to_csv_and_pgn(last_stable_state, stable_state, pgn_recorder)
            last_stable_state = stable_state.copy()
        
        # Use cached display update
        display_state = state_buffer.current_stable_state if state_buffer.current_stable_state else current_positions
        board_vis = chess_display.update(display_state)
        
        # Position the board window
        cv2.namedWindow('Chess Board', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Chess Board', 800, 800)
        cv2.moveWindow('Chess Board', 600, 0)
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
                    confidence = float(box.conf[0])
                    
                    # Draw bounding box and label with confidence
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
                    label_text = f"{piece_label} ({confidence:.2f})"
                    cv2.putText(annotated_frame, label_text, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
                    
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
        # Add FPS counter
        current_fps = 1.0 / frame_time
        cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", 
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Add inference time
        cv2.putText(annotated_frame, f"Inference: {inference_time*1000:.1f}ms", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Add uptime if resource monitor is enabled
        if resource_monitor:
            uptime = perf_counter() - resource_monitor.start_time
            cv2.putText(annotated_frame, f"Uptime: {int(uptime)}s", 
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('Live', annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        finalize_game(pgn_recorder)
        if resource_monitor:
            resource_monitor.stop()
        break
    elif key == ord('r'):
        print("Recalibrating grid...")
        state_buffer = reset_calibration()
        cv2.destroyWindow('Chess Board')  # Close the chess board window

cap.release()
cv2.destroyAllWindows()
if resource_monitor:  # Only stop if monitoring was enabled
    resource_monitor.stop()