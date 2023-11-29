import cv2

from robot import *


def image_info(img: np.ndarray):
    print(
        f'dtype: {img.dtype}, shape: {img.shape}, min: {img.min()}, max: {img.max()}')


def transform_perspective(matrix, point):
    transformed = matrix @ np.array([point[0], point[1], 1])
    return transformed[0:2] / transformed[2]


def transform_affine(matrix, point):
    transformed = matrix @ np.array([point[0], point[1], 1])
    return transformed


def get_image_to_world(width_px, height_px, width_mm, height_mm):
    src = np.float32([[0, 0], [width_px, 0], [0, height_px]])
    dst = np.float32([[0, height_mm], [width_mm, height_mm], [0, 0]])
    return cv2.getAffineTransform(src, dst)


def get_world_to_image(width_mm, height_mm, width_px, height_px):
    src = np.float32([[0, height_mm], [width_mm, height_mm], [0, 0]])
    dst = np.float32([[0, 0], [width_px, 0], [0, height_px]])
    return cv2.getAffineTransform(src, dst)


def get_perspective_transform(map_vertices: np.ndarray, dst_width, dst_height):
    pts_src = np.float32(map_vertices)
    sorted_by_y = np.argsort(pts_src[:, 1])
    top_points = pts_src[sorted_by_y[:2]]
    top_sorted_by_x = np.argsort(top_points[:, 0])
    top_left = top_points[top_sorted_by_x[0]]
    top_right = top_points[top_sorted_by_x[1]]
    bottom_points = pts_src[sorted_by_y[2:]]
    bottom_sorted_by_x = np.argsort(bottom_points[:, 0])
    bottom_left = bottom_points[bottom_sorted_by_x[0]]
    bottom_right = bottom_points[bottom_sorted_by_x[1]]
    pts_src = np.float32([top_left, top_right, bottom_right, bottom_left])
    pts_dst = np.float32([[0, 0], [dst_width, 0], [dst_width, dst_height], [0, dst_height]])
    return cv2.getPerspectiveTransform(pts_src, pts_dst)


def correct_perspective():
    img = cv2.imread('../images/perspective_box.jpg')
    assert img is not None
    dsize = (img.shape[1] // 8, img.shape[0] // 8)
    img = cv2.resize(img, dsize)

    pts_src = np.float32([[226, 173], [408, 273], [78, 275], [258, 424]])
    dst_width = 146 * 4
    dst_height = 126 * 4
    pts_dst = np.float32([[0, 0], [dst_width, 0], [0, dst_height], [dst_width, dst_height]])
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    inv_matrix = np.linalg.inv(matrix)

    for pt in pts_src:
        cv2.circle(img, center=pt.astype(int), radius=5, color=(0, 0, 255), thickness=-1)
    center = transform_perspective(inv_matrix, [dst_width / 3, dst_height / 3]).astype(int)
    cv2.circle(img, center=center, radius=5, color=(255, 0, 0), thickness=-1)
    cv2.imshow('main', img)
    cv2.waitKey(0)

    img = cv2.warpPerspective(img, matrix, (dst_width, dst_height))
    cv2.imshow('main', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def reconstruct_thymio():
    img = np.zeros((400, 400, 3), dtype=np.uint8)

    width_mm = 500
    height_mm = 500
    image_to_world = get_image_to_world(img.shape[1], img.shape[0], width_mm, height_mm)
    world_to_image = get_world_to_image(width_mm, height_mm, img.shape[1], img.shape[0])

    back = np.int32([200, 140])
    front_left = np.int32([180, 280])
    front_right = np.int32([300, 250])
    front_center = (front_left + front_right) // 2
    direction = front_center - back

    cv2.circle(img, center=back, radius=5, color=(0, 255, 0), thickness=-1)
    cv2.circle(img, center=front_left, radius=5, color=(0, 255, 0), thickness=-1)
    cv2.circle(img, center=front_right, radius=5, color=(0, 255, 0), thickness=-1)

    thymio = Thymio(back[0], back[1], np.arctan2(direction[1], direction[0]))

    robot_outline = thymio.get_outline()
    robot_outline = np.array([robot_outline.T[0, :], robot_outline.T[1, :], np.ones(robot_outline.shape[0])])
    robot_outline = (world_to_image @ robot_outline).T.astype(np.int32)
    robot_pos = (world_to_image @ np.array([thymio.pos_x, thymio.pos_y, 1])).astype(np.int32)

    cv2.polylines(img, [robot_outline], isClosed=True, color=(255, 255, 255), lineType=cv2.LINE_AA)
    cv2.circle(img, center=robot_pos, radius=3, color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

    cv2.imshow('main', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_obstacle_mask(color_image):
    """
    Returns a binary obstacle mask of the given image, where 1 represents an obstacle. An obstacle border is added.
    """

    threshold = 100
    kernel_size = 50
    img = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY_INV)

    # Create borders
    img[:, 0] = 1
    img[:, -1] = 1
    img[0, :] = 1
    img[-1, :] = 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    img = cv2.dilate(img, kernel)

    return img


def test_transforms():
    # Transformations we need:
    # - original image -> perspective corrected
    # -     detect the thymio in that space: (x, y, theta) in image space (left-handed system!)
    # - perspective corrected image space -> world space: simple affine transform (pixels -> mm)
    # - use the world space for everything else (graph, thymio position to send to filter)
    # Well, actually it would be much better to build the graph in image space
    #  -> integer coordinates -> exact equality tests

    width_px = 800
    height_px = 800
    width_mm = 1000
    height_mm = 1000

    # NOTE: this switches from left-handed to right-handed, maybe too confusing
    M = get_image_to_world(width_px, height_px, width_mm, height_mm)

    def print_transform(matrix, pt):
        print(f'{pt} -> {transform_affine(matrix, pt)}')

    print_transform(M, (0, 0))
    print_transform(M, (width_px, 0))
    print_transform(M, (0, height_px))
    print_transform(M, (width_px, height_px))
    print_transform(M, (width_px // 4, height_px // 4))


def draw_contour_orientations(img, contours, orientations):
    """
    Draw positive orientation as green, negative as red;
    first vertex is black, last is white
    """
    img[:] = (192, 192, 192)
    for c in range(len(contours)):
        color = (64, 192, 64) if orientations[c] >= 0 else (64, 64, 192)
        cv2.drawContours(img, [contours[c]], contourIdx=-1, color=color, thickness=3)
        n_points = len(contours[c])
        for i in range(n_points):
            brightness = i / (n_points - 1) * 255
            cv2.circle(img, contours[c][i], color=(brightness, brightness, brightness), radius=5, thickness=-1)


def detect_robot_vertices(hsv: cv2.typing.MatLike):
    lower_green = np.array([55, 50, 50])
    upper_green = np.array([75, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 3:
        return None

    vertices = np.empty((len(contours), 2), dtype=np.int32)
    for i in range(len(contours)):
        moments = cv2.moments(contours[i])
        if moments["m00"] == 0:
            return None

        px = np.int32(moments["m10"] / moments["m00"])
        py = np.int32(moments["m01"] / moments["m00"])
        vertices[i] = np.array([px, py])

    return vertices


# TODO: look at https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html !!!
#  That might be 10x more robust
def get_robot_pose(robot_vertices: np.ndarray, distance_center_back: float):
    """
    Compute the robot pose (position, direction) in image space from its three detected markers
    """

    lengths = np.empty(3)
    for i in range(len(robot_vertices)):
        vertex_1 = robot_vertices[i]
        vertex_2 = robot_vertices[(i + 1) % len(robot_vertices)]
        lengths[i] = np.linalg.norm(vertex_2 - vertex_1)

    # The front edge is the smallest of the three
    front_edge_index = np.argmin(lengths)

    # For now assume the left vertex comes before the right vertex
    left_index = front_edge_index
    right_index = (front_edge_index + 1) % len(robot_vertices)
    back_index = (front_edge_index + 2) % len(robot_vertices)
    left_edge = robot_vertices[left_index] - robot_vertices[back_index]
    right_edge = robot_vertices[right_index] - robot_vertices[back_index]
    # Switch left and right if we realize we were wrong
    if np.cross(left_edge, right_edge) < 0:
        left_index, right_index = right_index, left_index

    back = robot_vertices[back_index]
    left = robot_vertices[left_index]
    right = robot_vertices[right_index]
    front_center = (left + right) / 2
    direction = front_center - back
    direction /= np.linalg.norm(direction)
    position = back + distance_center_back * direction

    return position, direction


def detect_target(hsv: cv2.typing.MatLike):
    lower_pink = np.array([165, 50, 50])
    upper_pink = np.array([175, 255, 255])
    mask = cv2.inRange(hsv, lower_pink, upper_pink)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 1:
        return None

    moments = cv2.moments(contours[0])
    if moments["m00"] == 0:
        return None

    px = np.int32(moments["m10"] / moments["m00"])
    py = np.int32(moments["m01"] / moments["m00"])
    return np.array([px, py])


def detect_map_corners(hsv: cv2.typing.MatLike):
    lower_blue = np.array([100, 100, 50])
    upper_blue = np.array([115, 200, 200])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 4:
        return None

    vertices = np.empty((len(contours), 2), dtype=np.int32)
    for i in range(len(contours)):
        moments = cv2.moments(contours[i])
        if moments["m00"] == 0:
            return None

        px = np.int32(moments["m10"] / moments["m00"])
        py = np.int32(moments["m01"] / moments["m00"])
        vertices[i] = np.array([px, py])

    return vertices


if __name__ == '__main__':
    # correct_perspective()
    # reconstruct_thymio()
    # test_transforms()
    # test_obstacle_mask()
    pass
