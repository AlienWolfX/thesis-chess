import re
import cv2
from utils.chessFunctions import (
    fen_to_image,
    read_img,
    canny_edge,
    atoi,
    hough_line
)

def rescale_frame(frame, percent=75):
    """
    Rescale the given frame to the specified width and height.
    """
    dim = (1000, 750)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def natural_keys(text):
    """
    Sort strings containing numbers in a natural order.
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]

# IP CAM
cap = cv2.VideoCapture('http://192.168.6.206:8080/video')

# Select the live video stream source (0-webcam & 1-external camera)
# cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Could not open video device")

start = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'
board = fen_to_image(start)
board_image = cv2.imread('current_board.png')
if board_image is None:
    raise Exception("Could not read current_board.png")
cv2.imshow('current board', board_image)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    small_frame = rescale_frame(frame)
    cv2.imshow('live', small_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        print('Detecting...')
        cv2.imwrite('frame.jpeg', frame)
        
        img, gray_blur = read_img('frame.jpeg')
        cv2.imwrite("gray.jpeg", gray_blur)
        
        edges = canny_edge(gray_blur)
        cv2.imwrite("edges.jpeg", edges)
        
        lines = hough_line(edges)
        cv2.imwrite("lines.jpeg", lines)
                
        print(board)
        board_image = cv2.imread('current_board.png')
        if board_image is None:
            raise Exception("Could not read current_board.png")
        cv2.imshow('current board', board_image)
        print('Completed!')

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()