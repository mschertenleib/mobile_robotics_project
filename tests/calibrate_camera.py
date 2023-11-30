import cv2
import numpy as np


def main():
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

    ret_val, camera_matrix, distortion_coeffs, r_vecs, t_vecs = cv2.calibrateCamera(all_object_points, all_image_points,
                                                                              imageSize=(640, 480),
                                                                              cameraMatrix=np.empty((3, 3)),
                                                                              distCoeffs=np.empty(14))
    print(camera_matrix)
    print(distortion_coeffs)


if __name__ == '__main__':
    main()
