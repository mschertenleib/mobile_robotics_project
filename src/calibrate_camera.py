import json

import cv2
import numpy as np

from parameters import FRAME_WIDTH, FRAME_HEIGHT


def create_board() -> cv2.aruco.CharucoBoard:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    return cv2.aruco.CharucoBoard(size=(5, 7), squareLength=39.1, markerLength=19.6, dictionary=dictionary)


def generate_board_image(filename):
    board = create_board()
    board_image = board.generateImage(outSize=(1000, 1400), marginSize=20, borderBits=1)
    cv2.imwrite(filename, board_image)


def store_camera_to_json(filename, camera_matrix: np.ndarray, distortion_coeffs: np.ndarray):
    with open(filename, 'w') as f:
        data = {'camera_matrix': camera_matrix.tolist(),
                'distortion_coeffs': distortion_coeffs.squeeze().tolist()}
        json.dump(data, f)


def load_camera_from_json(filename) -> tuple[np.ndarray, np.ndarray]:
    with open(filename, 'r') as f:
        data = json.load(f)
        camera_matrix = np.array(data['camera_matrix'])
        distortion_coeffs = np.array(data['distortion_coeffs'])
        return camera_matrix, distortion_coeffs


def calibrate_camera(frame_width, frame_height):
    board = create_board()

    charuco_params = cv2.aruco.CharucoParameters()
    detector_params = cv2.aruco.DetectorParameters()
    refine_params = cv2.aruco.RefineParameters()
    detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params, refine_params)

    all_object_points = []
    all_image_points = []

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    if not cap.isOpened():
        print('Failed to open video capture')
        cap.release()
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('Cannot read frame')
            break

        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(frame)

        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)

        cv2.imshow('Calibration', frame)
        key = cv2.waitKey(1)
        if key & 0xff == 27:
            # If Esc is pressed, exit this loop and move on with the actual calibration computations
            break
        elif key >= 0 and charuco_ids is not None and len(charuco_ids) > 0:
            # If any key is pressed and some markers were detected, try to match image points and store them in the
            # global list
            object_points, image_points = board.matchImagePoints(charuco_corners, charuco_ids)
            if len(object_points) >= 4:
                all_object_points.append(object_points)
                all_image_points.append(image_points)

    cv2.destroyAllWindows()

    ret_val, camera_matrix, distortion_coeffs, r_vecs, t_vecs = cv2.calibrateCamera(all_object_points, all_image_points,
                                                                                    imageSize=(
                                                                                        frame_width, frame_height),
                                                                                    cameraMatrix=np.empty((3, 3)),
                                                                                    distCoeffs=np.empty(14))

    store_camera_to_json('./camera.json', camera_matrix, distortion_coeffs)

    # Simply show the undistorted camera frame until the user presses Esc, to assess if the calibration seems correct
    # or not
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('Cannot read frame')
            break
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs,
                                                               (frame_width, frame_height), 0,
                                                               (frame_width, frame_height))

        undistorted = cv2.undistort(frame, camera_matrix, distortion_coeffs, None, new_camera_matrix)

        cv2.imshow('Undistorted', undistorted)
        if cv2.waitKey(1) & 0xff == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # generate_board_image('charuco_board.png')
    calibrate_camera(FRAME_WIDTH, FRAME_HEIGHT)
