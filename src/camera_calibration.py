import json

import cv2
import numpy as np


def calibrate_camera(frame_width, frame_height):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    board = cv2.aruco.CharucoBoard(size=(5, 7), squareLength=39.1, markerLength=19.6, dictionary=dictionary)
    # board_image = board.generateImage(outSize=(1000, 1400), marginSize=20, borderBits=1)
    # cv2.imwrite('charuco_board.png', board_image)
    # return

    charuco_params = cv2.aruco.CharucoParameters()
    detector_params = cv2.aruco.DetectorParameters()
    refine_params = cv2.aruco.RefineParameters()
    detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params, refine_params)

    all_object_points = []
    all_image_points = []

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('Cannot read frame')
            break

        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(frame)

        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)

        cv2.imshow("main", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key >= 0 and charuco_ids is not None and len(charuco_ids) > 0:
            object_points, image_points = board.matchImagePoints(charuco_corners, charuco_ids)
            if len(object_points) >= 4:
                all_object_points.append(object_points)
                all_image_points.append(image_points)

        if key & 0xff == ord('s'):
            cv2.imwrite('test.png', frame)

    cap.release()
    cv2.destroyAllWindows()

    ret_val, camera_matrix, distortion_coeffs, r_vecs, t_vecs = cv2.calibrateCamera(all_object_points, all_image_points,
                                                                                    imageSize=(
                                                                                        frame_width, frame_height),
                                                                                    cameraMatrix=np.empty((3, 3)),
                                                                                    distCoeffs=np.empty(14))

    return camera_matrix, distortion_coeffs


def store_to_json(filename, camera_matrix: np.ndarray, distortion_coeffs: np.ndarray):
    with open(filename, 'w') as f:
        data = {'camera_matrix': camera_matrix.tolist(),
                'distortion_coeffs': distortion_coeffs.squeeze().tolist()}
        json.dump(data, f)


def load_from_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
        camera_matrix = np.array(data['camera_matrix'])
        distortion_coeffs = np.array(data['distortion_coeffs'])
        return camera_matrix, distortion_coeffs
