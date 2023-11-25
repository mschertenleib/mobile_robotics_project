from image_processing import *

mouse_x, mouse_y = 0, 0


def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y


def detect_robot():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('Cannot read frame')
            break

        img = cv2.GaussianBlur(frame, (15, 15), 0)
        # img = cv2.medianBlur(frame, 15)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_red = np.array([60, 50, 50])
        upper_red = np.array([80, 255, 255])

        global mouse_x, mouse_y
        print(hsv[mouse_y, mouse_x])

        mask = cv2.inRange(hsv, lower_red, upper_red)

        cv2.imshow('color', img)
        cv2.imshow('hue', hsv[:, :, 0])
        cv2.setMouseCallback('hue', mouse_callback)
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
    detect_map()
    # detect_robot()
