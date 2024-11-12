import re
import cv2
from utils.chessFunctions import (
    fen_to_image,
    read_img,
    canny_edge,
    atoi,
)

def rescale_frame(frame, percent=75):
    # width = int(frame.shape[1] * (percent / 100))
    # height = int(frame.shape[0] * (percent / 100))
    dim = (1000, 750)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


# Select the live video stream source (0-webcam & 1-external camera)
cap = cv2.VideoCapture(0)

# Select the IP address of the phone
# ip_address = ''
# cap = cv2.VideoCapture('http://{ip_address}:8080/video')


start = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'
board = fen_to_image(start)
board_image = cv2.imread('current_board.png')
cv2.imshow('current board', board_image)

while(True):
    ret, frame = cap.read()
    small_frame = rescale_frame(frame)
    cv2.imshow('live', small_frame)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        print('Detecting...')
        cv2.imwrite('frame.jpeg', frame)
        
        # # Low-level CV techniques (grayscale & blur)
        img, gray_blur = read_img('frame.jpeg')
        cv2.imwrite("gray.jpeg", gray_blur)
        
        # # Canny algorithm
        edges = canny_edge(gray_blur)
        cv2.imwrite("edges.jpeg", edges)
                
        print(board)
        board_image = cv2.imread('current_board.png')
        cv2.imshow('current board', board_image)
        print('Completed!')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()