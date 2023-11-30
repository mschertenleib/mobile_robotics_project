import cv2

import numpy as np


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

        corners, ids, rejected = detector.detectMarkers(frame)

        img = frame.copy()
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
            for i in range(len(ids)):
                cv2.circle(img, center=corners[i][0, 0].astype(np.int32), radius=5, color=(0, 0, 255), thickness=-1)
                cv2.circle(img, center=corners[i][0, 1].astype(np.int32), radius=5, color=(0, 255, 0), thickness=-1)
                cv2.circle(img, center=corners[i][0, 2].astype(np.int32), radius=5, color=(255, 0, 0), thickness=-1)
                cv2.circle(img, center=corners[i][0, 3].astype(np.int32), radius=5, color=(0, 255, 255), thickness=-1)

        cv2.imshow("main", img)
        key = cv2.waitKey(1) & 0xff
        if key == 27:
            break


if __name__ == '__main__':
    main()
