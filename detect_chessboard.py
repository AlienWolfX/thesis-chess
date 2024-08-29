import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def select_points(event, x, y, flags, param):
    image, selected_points = param
    if event == cv.EVENT_LBUTTONDOWN:
        selected_points.append((x, y))
        cv.circle(image, (x, y), 5, (0, 255, 0), -1)
        if len(selected_points) > 1:
            cv.line(image, selected_points[-2], selected_points[-1], (0, 255, 0), 2)
        if len(selected_points) == 4:
            cv.line(image, selected_points[3], selected_points[0], (0, 255, 0), 2)
        cv.imshow("Select Corners", image)

def calibrate_chessboard(image):
    selected_points = []
    cv.imshow("Select Corners", image)
    cv.setMouseCallback("Select Corners", select_points, (image, selected_points))
    cv.waitKey(0)
    cv.destroyWindow("Select Corners")
    if len(selected_points) != 4:
        raise ValueError("Exactly 4 points must be selected.")
    return selected_points

def flatten_chessboard(image, corners):
    width, height = 400, 400
    dst_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    src_points = np.array(corners, dtype=np.float32)
    matrix = cv.getPerspectiveTransform(src_points, dst_points)
    flattened_image = cv.warpPerspective(image, matrix, (width, height))
    return flattened_image

def extract_tiles(flattened_image):
    tiles = []
    tile_size = flattened_image.shape[0] // 8
    for row in range(8):
        for col in range(8):
            x_start = col * tile_size
            y_start = row * tile_size
            tile = flattened_image[y_start:y_start + tile_size, x_start:x_start + tile_size]
            tiles.append(tile)
    return tiles

def process_image(src_image_path, canny_low_threshold=50, canny_high_threshold=150):
    src_image = cv.imread(src_image_path)
    if src_image is None:
        raise FileNotFoundError(f"Image not found at path: {src_image_path}")

    corners = calibrate_chessboard(src_image.copy())
    flattened_image = flatten_chessboard(src_image, corners)
    denoised_image = cv.fastNlMeansDenoisingColored(flattened_image, None, 10, 10, 10, 21)
    tiles = extract_tiles(denoised_image)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv.cvtColor(src_image, cv.COLOR_BGR2RGB))
    plt.title('Original Image')

    plt.figure(figsize=(10, 10))
    for i, tile in enumerate(tiles):
        plt.subplot(8, 8, i + 1)
        plt.imshow(cv.cvtColor(tile, cv.COLOR_BGR2RGB))
        plt.axis('off')
    plt.suptitle('Separated Tiles')
    plt.show()

def main():
    src_image_path = "img/chess.jpg"
    try:
        process_image(src_image_path)
    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()