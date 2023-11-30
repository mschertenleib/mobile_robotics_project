import cv2


def main():
    n_markers = 6
    dictionary = cv2.aruco.extendDictionary(nMarkers=n_markers, markerSize=6)
    for i in range(n_markers):
        marker_image = cv2.aruco.generateImageMarker(dictionary, id=i, sidePixels=280, borderBits=1)
        cv2.imwrite(f'aruco_marker_{i}.png', marker_image)


if __name__ == '__main__':
    main()
