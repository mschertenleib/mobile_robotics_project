from typing import *

import cv2

from parameters import *


def transform_perspective(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    transformed = matrix @ np.array([point[0], point[1], 1])
    return transformed[0:2] / transformed[2]


def transform_affine(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    transformed = matrix @ np.array([point[0], point[1], 1])
    return transformed


def get_image_to_world_matrix(width_px: int, height_px: int, width_mm: float, height_mm: float) -> np.ndarray:
    src = np.float32([[0, 0], [width_px, 0], [0, height_px]])
    dst = np.float32([[0, height_mm], [width_mm, height_mm], [0, 0]])
    return cv2.getAffineTransform(src, dst)


def get_world_to_image_matrix(width_mm: float, height_mm: float, width_px: int, height_px: int) -> np.ndarray:
    src = np.float32([[0, height_mm], [width_mm, height_mm], [0, 0]])
    dst = np.float32([[0, 0], [width_px, 0], [0, height_px]])
    return cv2.getAffineTransform(src, dst)


def get_perspective_transform(map_corners: np.ndarray, dst_width: int, dst_height: int) -> np.ndarray:
    pts_src = map_corners.astype(np.float32)
    pts_dst = np.float32([[0, 0], [dst_width, 0], [dst_width, dst_height], [0, dst_height]])
    return cv2.getPerspectiveTransform(pts_src, pts_dst)


def get_obstacle_mask(img: np.ndarray, robot_position: Optional[np.ndarray],
                      target_position: Optional[np.ndarray]) -> np.ndarray:
    """
    Returns a binary obstacle mask of the given color image, where 1 represents an obstacle.
    A border is also added. The four corner markers are not considered obstacles, as well as the robot and the target,
    if their position is not None.
    """

    threshold = 150
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY_INV)

    # Mask out the robot, target and corner markers, in order to not consider them as obstacles
    if robot_position is not None:
        cv2.circle(img, center=robot_position.astype(np.int32), radius=ROBOT_RADIUS_PX, color=[0], thickness=-1)
    if target_position is not None:
        cv2.circle(img, center=target_position.astype(np.int32), radius=TARGET_RADIUS_PX, color=[0], thickness=-1)
    cv2.rectangle(img, pt1=(0, 0), pt2=(MARKER_SIZE_PX, MARKER_SIZE_PX), color=[0], thickness=-1)
    cv2.rectangle(img, pt1=(MAP_WIDTH_PX - MARKER_SIZE_PX, 0), pt2=(MAP_WIDTH_PX, MARKER_SIZE_PX), color=[0],
                  thickness=-1)
    cv2.rectangle(img, pt1=(0, MAP_HEIGHT_PX - MARKER_SIZE_PX), pt2=(MARKER_SIZE_PX, MAP_HEIGHT_PX), color=[0],
                  thickness=-1)
    cv2.rectangle(img, pt1=(MAP_WIDTH_PX - MARKER_SIZE_PX, MAP_HEIGHT_PX - MARKER_SIZE_PX),
                  pt2=(MAP_WIDTH_PX, MAP_HEIGHT_PX), color=[0], thickness=-1)

    # Filter isolated obstacle pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, dst=img, iterations=1)

    # Create borders
    img[:, 0] = 1
    img[:, -1] = 1
    img[0, :] = 1
    img[-1, :] = 1

    dilation_kernel_size = 2 * DILATION_RADIUS_PX + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
    img = cv2.dilate(img, kernel)

    return img


def detect_map(marker_corners, marker_ids) -> tuple[bool, np.ndarray]:
    """
    Returns whether all map corners were detected and their position in image space.
    The corners are ordered clockwise, starting from the top left.
    """

    # NOTE: marker_corners and marker_ids are assumed to be directly obtained from
    # cv2.aruco.ArucoDetector.detectMarkers().
    # marker_corners is a tuple of arrays of shape (1, 4, 2)
    # marker_ids is a tuple of arrays of shape (1)

    if marker_ids is None:
        return False, np.array([])

    marker_indices = []
    for i in range(4):
        index = np.argwhere(np.array(marker_ids).flatten() == i)
        if len(index) == 1:
            marker_indices.append(index.item(0, 0))

    all_found = len(marker_indices) == 4
    corners = np.array(marker_corners)[marker_indices, :, 0, 0]

    return all_found, corners


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
