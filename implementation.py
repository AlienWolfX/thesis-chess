import cv2
import torch
from ultralytics import YOLO
import time

from utils.chessFunctions import fen_to_image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = YOLO('model/v9.pt').to(device)

# Initialize the webcam or IP camera
# cameraSrc = "https://192.168.6.207:8080/video" # IP Camera URL
cameraSrc = 0  # Laptop Webcam

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

# Display the starting board
start = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'
board = fen_to_image(start)
board_image = cv2.imread('current_board.png')

# Dictionary to store the original points of rooks
whiteRookPoint = []
blackRookPoint = []

# Detect initial positions of the rooks
start_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (320, 240))  
    
    results = model(resized_frame)
    print(results)

    detected_rook = False
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            center_bottom_x = (x1 + x2) // 2
            center_bottom_y = y2
            
            fen_label = class_id_mapping.get(model.names[int(box.cls)], '')
            
            if fen_label == 'R':
                whiteRookPoint.append((center_bottom_x, center_bottom_y))
                detected_rook = True
            elif fen_label == 'r':
                blackRookPoint.append((center_bottom_x, center_bottom_y))
                detected_rook = True

    if detected_rook:
        start_time = time.time()

    if len(whiteRookPoint) == 2 and len(blackRookPoint) == 2:
        break

    if time.time() - start_time == 2:
        print("No rooks detected")
        break

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    resized_frame = cv2.resize(frame, (320, 240))  
    
    results = model(resized_frame)
    print(results)

    annotated_frame = resized_frame.copy()

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            center_bottom_x = (x1 + x2) // 2
            center_bottom_y = y2
            cv2.circle(annotated_frame, (center_bottom_x, center_bottom_y), 3, (255, 255, 255), -1)
            
            fen_label = class_id_mapping.get(model.names[int(box.cls)], '')
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
            
            cv2.putText(annotated_frame, fen_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if len(whiteRookPoint) == 2 and len(blackRookPoint) == 2:
        cv2.line(annotated_frame, whiteRookPoint[0], blackRookPoint[0], (255, 0, 0), 1)
        cv2.line(annotated_frame, whiteRookPoint[1], blackRookPoint[1], (255, 0, 0), 1)

    # Display the annotated frame
    cv2.imshow('Live Inference', annotated_frame)

    # Display the chess board image
    if board_image is not None:
        cv2.imshow('Current Board', board_image)
    else:
        print("Error: Could not load the chess board image.")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()