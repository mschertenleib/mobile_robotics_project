from image_processing import *


mouse_x, mouse_y = 0, 0


def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('Cannot read frame')
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red = np.array([165, 100, 100])
        upper_red = np.array([179, 255, 255])

        hsv = cv2.GaussianBlur(hsv, (15, 15), 0)
        global mouse_x, mouse_y
        print(hsv[mouse_y, mouse_x])

        mask = cv2.inRange(hsv, lower_red, upper_red)

        cv2.imshow('frame', hsv)
        cv2.setMouseCallback('frame', mouse_callback)
        cv2.imshow('mask', mask)

        if cv2.waitKey(1) & 0xff == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
