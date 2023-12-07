from global_map import *
from image_processing import *

g_target = (120, 730)
g_source = (200, 100)


def draw_contour_orientations(img: np.ndarray, contours: list[np.ndarray]):
    """
    Draw positive orientation as green, negative as red;
    first vertex is black, last is white
    """
    for c in range(len(contours)):
        orientation = np.sign(cv2.contourArea(contours[c], oriented=True))
        color = (64, 192, 64) if orientation >= 0 else (64, 64, 192)
        cv2.drawContours(img, [contours[c]], contourIdx=-1, color=color, thickness=2)
        n_points = len(contours[c])
        for i in range(n_points):
            brightness = i / (n_points - 1) * 255
            cv2.circle(img, contours[c][i], color=(brightness, brightness, brightness), radius=5, thickness=-1)


def main():
    color_image = cv2.imread('../images/map_divided.png')
    obstacle_mask = get_obstacle_mask(color_image, robot_position=None, target_position=None, mask_corner_markers=False)

    approx_poly_epsilon = 2
    regions = extract_contours(obstacle_mask, approx_poly_epsilon)

    all_contours = [contour for region in regions for contour in region]

    free_space = np.empty_like(color_image)
    free_space[:] = (64, 64, 192)
    cv2.drawContours(free_space, all_contours, contourIdx=-1, color=(255, 255, 255), thickness=-1)

    graph = build_graph(regions)

    cv2.namedWindow('main', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('main', color_image.shape[1], color_image.shape[0])

    while True:
        free_source, free_target = update_graph(graph, regions, np.array(g_source), np.array(g_target))
        path = dijkstra(graph.adjacency, Graph.SOURCE, Graph.TARGET)

        img = cv2.addWeighted(color_image, 0.75, free_space, 0.25, 0.0)
        draw_contour_orientations(img, all_contours)
        cv2.drawContours(img, all_contours, contourIdx=-1, color=(64, 64, 192))
        draw_graph(img, graph)
        draw_path(img, graph, path, np.array(g_source), free_source, np.array(g_target), free_target)

        cv2.namedWindow('main', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('main', mouse_callback)
        cv2.imshow('main', img)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()


def mouse_callback(event, x, y, flags, param):
    global g_target, g_source
    if event == cv2.EVENT_MOUSEMOVE:
        g_target = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        g_source = (x, y)


if __name__ == '__main__':
    main()
