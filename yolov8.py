import cv2
import torch
from ultralytics import YOLO

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the YOLOv8 model
model = YOLO('model/50_epoch_batch_64.pt').to(device)

# Initialize the webcam ip or usb
# cameraSrc = "https://{ip}/video" # IP Camera URL
cameraSrc = 0 # Laptop Webcam

cap = cv2.VideoCapture(cameraSrc)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    results = model(frame)

    # Display results
    annotated_frame = results[0].plot()
    cv2.imshow('YOLOv8 Inference', annotated_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()