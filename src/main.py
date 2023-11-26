from image_processing import *

g_mouse_x, g_mouse_y = 0, 0


def mouse_callback(event, x, y, flags, param):
    global g_mouse_x, g_mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        g_mouse_x, g_mouse_y = x, y


def print_pixel(img, x, y):
    bgr = img[y, x]
    hsv = cv2.cvtColor(np.array([[bgr]]), code=cv2.COLOR_BGR2HSV).flatten()
    print(f'XY = ({x}, {y}), RGB = {bgr[::-1]}, HSV = {hsv}')


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('Cannot read frame')
            break

        img = frame.copy()
        vertices = detect_robot(img)
        if vertices is None:
            cv2.putText(img, 'Robot not detected', org=(10, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=(64, 64, 192), lineType=cv2.LINE_AA)
        else:
            cv2.polylines(img, [np.array(vertices)], isClosed=True, color=(0, 255, 0))
            for vertex in vertices:
                cv2.drawMarker(img, position=vertex, color=(0, 0, 255), markerType=cv2.MARKER_CROSS)

        cv2.imshow('img', img)
        cv2.setMouseCallback('img', mouse_callback)

        key = cv2.waitKey(1) & 0xff
        if key == 27:
            break
        elif key == ord('p'):
            print_pixel(frame, g_mouse_x, g_mouse_y)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
