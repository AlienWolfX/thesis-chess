import cv2 as cv
import numpy as np

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

def calibrate_chessboard(image, display_width=800, display_height=800):
    resized_image = cv.resize(image, (display_width, display_height), interpolation=cv.INTER_AREA)
    
    selected_points = []
    cv.imshow("Select Corners", resized_image)
    cv.setMouseCallback("Select Corners", select_points, (resized_image, selected_points))
    cv.waitKey(0)
    cv.destroyWindow("Select Corners")
    if len(selected_points) != 4:
        raise ValueError("Exactly 4 points must be selected.")
    
    scale_x = image.shape[1] / display_width
    scale_y = image.shape[0] / display_height
    original_points = [(int(x * scale_x), int(y * scale_y)) for x, y in selected_points]
    
    return original_points

def main():
    # Use 0 for the default camera, or replace with the IP camera URL
    camera_source = 0  # or "http://<ip_camera_url>/video"

    cap = cv.VideoCapture(camera_source)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    corners = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        if corners is None:
            cv.imshow("Camera Feed", frame)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('c'):
                corners = calibrate_chessboard(frame.copy())
        else:
            mask = np.zeros_like(frame)
            roi_corners = np.array([corners], dtype=np.int32)
            cv.fillPoly(mask, roi_corners, (255, 255, 255))
            masked_image = cv.bitwise_and(frame, mask)
            cv.imshow("Focused Area", masked_image)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                corners = None

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()