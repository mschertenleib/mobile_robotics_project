import cv2


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    detector_params = cv2.aruco.DetectorParameters()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    # dictionary = cv2.aruco.extendDictionary(nMarkers=6, markerSize=5)
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('Cannot read frame')
            break

        corners, ids, rejected = detector.detectMarkers(frame)

        img = frame.copy()
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(img, corners, ids)

        cv2.imshow("main", img)
        key = cv2.waitKey(1) & 0xff
        if key == 27:
            break


if __name__ == '__main__':
    main()
