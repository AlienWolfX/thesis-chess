import cv2
import torch
from ultralytics import YOLO

from utils.chessFunctions import fen_to_image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = YOLO('model/best.pt').to(device)

# Initialize the webcam or IP camera
# cameraSrc = "https://192.168.6.207:8080/video" # IP Camera URL
cameraSrc = 0  # Laptop Webcam

cap = cv2.VideoCapture(cameraSrc)

frame_skip = 2 
frame_count = 0

# Display the starting board
start = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'
board = fen_to_image(start)
board_image = cv2.imread('current_board.png')

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

    annotated_frame = results[0].plot(line_width=1) 

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            center_bottom_x = (x1 + x2) // 2
            center_bottom_y = y2
            cv2.circle(annotated_frame, (center_bottom_x, center_bottom_y), 3, (255, 255, 255), -1)

    # print(board) # Current board still not implemented
    board_image = cv2.imread('current_board.png')
    cv2.imshow('current board', board_image)
    
    cv2.imshow('Live Inference', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
