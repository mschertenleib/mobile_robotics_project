from image_processing import *


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    detector_params = cv2.aruco.DetectorParameters()
    dictionary = cv2.aruco.extendDictionary(nMarkers=6, markerSize=6)
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('Cannot read frame')
            break

        img = frame.copy()

        corners, ids, rejected = detector.detectMarkers(img)

        robot_found, robot_position, robot_direction = detect_robot(corners, ids)
        target_found, target_position = detect_target(corners, ids)

        if robot_found:
            pos = robot_position.astype(np.int32)
            tip = (robot_position + robot_direction).astype(np.int32)
            cv2.line(img, pos, tip, color=(0, 0, 255))
            cv2.drawMarker(img, position=pos, color=(0, 0, 255), markerSize=10,
                           markerType=cv2.MARKER_CROSS)

        if target_found:
            cv2.drawMarker(img, position=target_position.astype(np.int32), color=(0, 255, 0), markerSize=10,
                           markerType=cv2.MARKER_CROSS)

        cv2.imshow("main", img)
        key = cv2.waitKey(1) & 0xff
        if key == 27:
            break


if __name__ == '__main__':
    main()
