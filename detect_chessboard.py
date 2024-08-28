"""
This file is used to detect the chessboard from the input.
"""

import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

def sobel(src_image, kernel_size):
    grad_x = cv.Sobel(src_image, cv.CV_16S, 1, 0, ksize=kernel_size, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(src_image, cv.CV_16S, 0, 1, ksize=kernel_size, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    return grad

def process_image(src_image_path, blur_kernel_size=(3, 3), sobel_kernel_size=3, harris_block_size=2, harris_ksize=3, harris_k=0.04):
    src_image = cv.imread(src_image_path)
    if src_image is None:
        raise FileNotFoundError(f"Image not found at path: {src_image_path}")
    
    src_image = cv.cvtColor(src_image, cv.COLOR_BGR2RGB)
    src_gray = cv.cvtColor(src_image, cv.COLOR_BGR2GRAY)

    blur_image = cv.blur(src_gray, blur_kernel_size)

    sobel_image = sobel(blur_image, sobel_kernel_size)

    corners = cv.cornerHarris(sobel_image, harris_block_size, harris_ksize, harris_k)
    corners = cv.dilate(corners, None)

    dest_image = np.copy(src_image)
    dest_image[corners > 0.01 * corners.max()] = [0, 0, 255]
    
    return dest_image 

def main():
    src_image_path = "img/board1.jpg"
    try:
        dest_image = process_image(src_image_path)
        plt.imshow(dest_image)
        plt.title("Detected Chessboard Corners")
        plt.axis('off')
        plt.show()
    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    main()