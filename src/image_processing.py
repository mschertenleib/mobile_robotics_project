import cv2

from robot import *


def image_info(img: np.ndarray):
    print(
        f'dtype: {img.dtype}, shape: {img.shape}, min: {img.min()}, max: {img.max()}')


def transform_perspective(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    transformed = matrix @ np.array([point[0], point[1], 1])
    return transformed[0:2] / transformed[2]


def transform_affine(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    transformed = matrix @ np.array([point[0], point[1], 1])
    return transformed


def get_image_to_world(width_px: int, height_px: int, width_mm: float, height_mm: float) -> np.ndarray:
    src = np.float32([[0, 0], [width_px, 0], [0, height_px]])
    dst = np.float32([[0, height_mm], [width_mm, height_mm], [0, 0]])
    return cv2.getAffineTransform(src, dst)


def get_world_to_image(width_mm: float, height_mm: float, width_px: int, height_px: int) -> np.ndarray:
    src = np.float32([[0, height_mm], [width_mm, height_mm], [0, 0]])
    dst = np.float32([[0, 0], [width_px, 0], [0, height_px]])
    return cv2.getAffineTransform(src, dst)


def get_perspective_transform(map_corners: np.ndarray, dst_width: int, dst_height: int) -> np.ndarray:
    pts_src = map_corners.astype(np.float32)
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


def get_obstacle_mask(color_image: np.ndarray) -> np.ndarray:
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



def draw_contour_orientations(img: np.ndarray, contours: list[np.ndarray], orientations: np.ndarray):
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


def detect_robot(marker_corners, marker_ids) -> tuple[bool, np.ndarray, np.ndarray]:
    """
    Returns whether the robot was detected, its position and its direction in image space
    """

    if marker_ids is not None and len(marker_ids) > 0:
        marker_index = np.argwhere(np.array(marker_ids).flatten() == 4)
        if np.size(marker_index) == 1:
            corners = np.array(marker_corners)[marker_index.flatten().item(0)].squeeze()
            position = np.sum(corners, 0) / 4
            direction = (corners[0] + corners[1]) / 2 - (corners[2] + corners[3]) / 2
            direction /= np.linalg.norm(direction)
            return True, position, direction

    return False, np.zeros(2), np.zeros(2)


def detect_target(marker_corners, marker_ids) -> tuple[bool, np.ndarray]:
    """
    Returns whether the target was detected and its position in image space
    """

    if marker_ids is not None and len(marker_ids) > 0:
        marker_index = np.argwhere(np.array(marker_ids).flatten() == 5)
        if np.size(marker_index) == 1:
            corners = np.array(marker_corners)[marker_index.flatten().item(0)].squeeze()
            position = np.sum(corners, 0) / 4
            return True, position

    return False, np.zeros(2)


def detect_map(marker_corners, marker_ids) -> tuple[bool, np.ndarray]:
    """
    Returns whether all map corners were detected and their position in image space.
    The corners are ordered clockwise, starting from the top left.
    """

    if marker_ids is not None and len(marker_ids) >= 4:
        marker_indices = []
        for i in range(4):
            index = np.argwhere(np.array(marker_ids) == i)
            if len(index) == 1:
                marker_indices.append(index.flatten().item(0))
        if len(marker_indices) == 4:
            corners = np.array(marker_corners)[marker_indices].squeeze()[:, 0]
            return True, corners

    return False, np.zeros(2)
