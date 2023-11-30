import json

import cv2
import numpy as np


def calibrate_camera():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    board = cv2.aruco.CharucoBoard(size=(5, 7), squareLength=0.04, markerLength=0.02, dictionary=dictionary)
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

    ret_val, camera_matrix, distortion_coeffs, r_vecs, t_vecs = cv2.calibrateCamera(all_object_points, all_image_points,
                                                                                    imageSize=(640, 480),
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
