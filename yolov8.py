import cv2
import torch
import numpy as np
from ultralytics import YOLO

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the YOLOv8 model
model = YOLO('model/v29.pt').to(device)

# Initialize the webcam ip or usb
# cameraSrc = "http://192.168.0.100:8080/video" # IP Camera URL
cameraSrc = 0 # Laptop Webcam

cap = cv2.VideoCapture(cameraSrc)

def order_points(pts):
    """Order points in clockwise order starting from top-left"""
    rect = np.zeros((4, 2), dtype="float32")
    
    # Top-left point has smallest sum
    # Bottom-right point has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Top-right point has smallest difference
    # Bottom-left point has largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def get_perspective_transform(frame, corners):
    """Get perspective transform matrix"""
    # Order corner points
    src_pts = order_points(np.array(corners, dtype="float32"))
    
    # Define destination points for 800x800 square
    width = height = 800
    dst_pts = np.array([
        [0, 0],
        [width-1, 0],
        [width-1, height-1],
        [0, height-1]
    ], dtype="float32")
    
    # Calculate perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Apply perspective transform
    warped = cv2.warpPerspective(frame, matrix, (width, height))
    
    return warped, matrix

# Track if corners are found
corners_found = False
corner_points = []

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    results = model(frame)
    
    # Get corner detections
    if not corners_found:
        corner_points = []
        for r in results:
            for det in r.boxes.data:
                cls = int(det[5])
                if cls == 0:  # Assuming class 0 is chessboard corner
                    x, y = int(det[0]), int(det[1])
                    corner_points.append([x, y])
        
        # If we found 4 corners, apply perspective transform
        if len(corner_points) == 4:
            corners_found = True
            warped_frame, transform_matrix = get_perspective_transform(frame, corner_points)
            cv2.imshow('Transformed Board', warped_frame)

    # Display results
    annotated_frame = results[0].plot()
    cv2.imshow('YOLOv8 Inference', annotated_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()