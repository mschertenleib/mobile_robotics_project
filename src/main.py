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
        text_y = 25

        robot_vertices = detect_robot(frame)
        if robot_vertices is None:
            cv2.putText(img, 'Robot not detected', org=(10, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                        color=(64, 64, 192), lineType=cv2.LINE_AA)
            text_y += 20
        else:
            cv2.polylines(img, [np.array(robot_vertices)], isClosed=True, color=(0, 255, 0))
            for vertex in robot_vertices:
                cv2.drawMarker(img, position=vertex, color=(0, 0, 255), markerType=cv2.MARKER_CROSS)

        map_vertices = detect_map(frame)
        if map_vertices is None:
            cv2.putText(img, 'Map not detected', org=(10, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                        color=(64, 64, 192), lineType=cv2.LINE_AA)
        else:
            cv2.drawContours(img, [map_vertices], contourIdx=-1, color=(0, 255, 0))
            for vertex in map_vertices:
                cv2.drawMarker(img, position=vertex[0], color=(0, 0, 255), markerType=cv2.MARKER_CROSS)

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
