import cv2

import numpy as np

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
        get_robot_pose(img, detector)

        cv2.imshow("main", img)
        key = cv2.waitKey(1) & 0xff
        if key == 27:
            break


if __name__ == '__main__':
    main()
