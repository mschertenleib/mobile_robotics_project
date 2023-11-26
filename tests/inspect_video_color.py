from image_processing import *

g_mouse_x, g_mouse_y = 0, 0


def mouse_callback(event, x, y, flags, param):
    global g_mouse_x, g_mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('Cannot read frame')
            break

        img = cv2.GaussianBlur(frame, (15, 15), 0)
        # img = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

        global g_mouse_x, g_mouse_y
        bgr = img[mouse_y, mouse_x]
        hsv = cv2.cvtColor(np.array([[bgr]]), cv2.COLOR_BGR2HSV).flatten()
        print(f'XY = ({mouse_x}, {mouse_y}), RGB = {bgr[::-1]}, HSV = {hsv}')

        cv2.imshow('main', img)
        cv2.setMouseCallback('main', mouse_callback)
        if cv2.waitKey(1) & 0xff == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
