from image_processing import *


def detect_robot():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('Cannot read frame')
            break

        img = frame.copy()
        img = cv2.bilateralFilter(img, 15, 150, 150)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([60, 50, 50])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        vertices = []
        for contour in contours:
            moments = cv2.moments(contour)
            px = int(moments["m10"] / moments["m00"])
            py = int(moments["m01"] / moments["m00"])
            vertices.append(np.array([px, py]))

        cv2.drawContours(img, contours, contourIdx=-1, color=(255, 255, 255))
        cv2.polylines(img, [np.array(vertices)], isClosed=True, color=(0, 255, 0))
        for vertex in vertices:
            cv2.drawMarker(img, position=vertex, color=(0, 0, 255), markerType=cv2.MARKER_CROSS)

        cv2.imshow('img', img)
        cv2.imshow('mask', mask)

        if cv2.waitKey(1) & 0xff == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_map():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('Cannot read frame')
            break

        img = frame.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(img, 200, 1, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)

        cv2.imshow('img', img * 255)
        if cv2.waitKey(1) & 0xff == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # detect_map()
    detect_robot()
