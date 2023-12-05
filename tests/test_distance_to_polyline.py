from global_map import *

g_point = (0, 0)
g_vertices = []


def mouse_callback(event, x, y, flags, param):
    global g_point, g_vertices
    if event == cv2.EVENT_MOUSEMOVE:
        g_point = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        g_vertices.append((x, y))


def main():
    width = 640
    height = 480

    global g_point, g_vertices

    while True:
        distance = distance_to_polyline(np.array(g_point, dtype=np.int32), np.array(g_vertices, dtype=np.int32))

        img = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)
        cv2.polylines(img, [np.array(g_vertices, dtype=np.int32)], isClosed=False, color=(0, 0, 0), thickness=2,
                      lineType=cv2.LINE_AA)
        cv2.putText(img, f'{distance}', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(0, 0, 0), lineType=cv2.LINE_AA)

        cv2.namedWindow('main', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('main', mouse_callback)
        cv2.imshow('main', img)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
