import cv2
import numpy as np

from camera_calibration import *


def test_calibration(img_filename, camera_matrix: np.ndarray, distortion_coeffs: np.ndarray):
    img = cv2.imread(img_filename)
    height, width = img.shape[:2]

    cv2.imshow('main', img)
    cv2.waitKey(0)

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs, (width, height), 0,
                                                           (width, height))

    undistorted = cv2.undistort(img, camera_matrix, distortion_coeffs, None, new_camera_matrix)
    cv2.imshow('main', undistorted)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


def main():
    camera_matrix, distortion_coeffs = calibrate_camera()
    json_filename = 'camera.json'
    store_to_json(json_filename, camera_matrix, distortion_coeffs)
    camera_matrix, distortion_coeffs = load_from_json(json_filename)
    test_calibration('../images/capture.png', camera_matrix, distortion_coeffs)


if __name__ == '__main__':
    main()
