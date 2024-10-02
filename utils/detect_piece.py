"""
This file is used to detect the chess piece from the input using YOLOv5.
"""
import cv2
import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace().project("chessv1-5ew7x")
model = project.version(1).model 

piece_class = {
    8: 'White King',
    12: 'White Rook',
    11: 'White Queen',
    7: 'White Bishop',
    10: 'White Pawn',
    9: 'White Knight',
    1: 'Black Bishop',
    3: 'Black Knight',
    5: 'Black Queen',
    4: 'Black Pawn',
    2: 'Black King',
    6: 'Black Rook'
}

def detectPiece(path):
    img = cv2.imread(path)

    predictions = model.predict(path, confidence=70, overlap=60).json()
    predictions = predictions['predictions']

    for prediction in predictions:
        x1, y1, x2, y2 = int(prediction['x'] - prediction['width'] / 2), int(prediction['y'] - prediction['height'] / 2), int(prediction['x'] + prediction['width'] / 2), int(prediction['y'] + prediction['height'] / 2)
        class_id = int(prediction['class'])
        label = f"{piece_class.get(class_id, 'Unknown')} {prediction['confidence']:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 20, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (94, 32, 25), 2)

    output_path = "prediction.jpg"
    cv2.imwrite(output_path, img)

path = "img/chess.jpg"
detectPiece(path)