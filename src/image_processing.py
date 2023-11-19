import cv2

from robot import *


def image_info(img: np.ndarray):
    print(
        f'dtype: {img.dtype}, shape: {img.shape}, min: {img.min()}, max: {img.max()}')


def normalize(img: np.ndarray):
    range_values = img.max() - img.min()
    if range_values == 0:
        return img.astype(np.float32)
    else:
        return (img.astype(np.float32) - img.min()) / range_values


def transform(matrix, point):
    transformed = matrix @ np.float32([point[0], point[1], 1])
    return transformed[0:2] / transformed[2]


def draw_thymio(img, position: np.ndarray, angle: float):
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    points = position.astype(float) + (rot @ Thymio.outline.T).T
    cv2.polylines(img, [points.astype(int)], isClosed=True, color=(255, 255, 255))


def correct_perspective():
    img = cv2.imread('../perspective_box.jpg')
    assert img is not None
    img = cv2.resize(img, (img.shape[1] // 8, img.shape[0] // 8))

    pts_src = np.float32([[226, 173], [408, 273], [78, 275], [258, 424]])
    dst_width = 146 * 4
    dst_height = 126 * 4
    pts_dst = np.float32([[0, 0], [dst_width, 0], [0, dst_height], [dst_width, dst_height]])
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    inv_matrix = np.linalg.inv(matrix)

    for pt in pts_src:
        cv2.circle(img, center=pt.astype(int), radius=5, color=(0, 0, 255), thickness=-1)
    center = transform(inv_matrix, [dst_width / 3, dst_height / 3]).astype(int)
    cv2.circle(img, center=center, radius=5, color=(255, 0, 0), thickness=-1)
    cv2.imshow('main', img)
    cv2.waitKey(0)

    img = cv2.warpPerspective(img, matrix, (dst_width, dst_height))
    cv2.imshow('main', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def reconstruct_thymio():
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    # TODO: convert to HSV for detecting dots? But might actually be worse

    tip = np.float32([200, 140])
    left = np.float32([180, 280])
    right = np.float32([300, 250])
    center = (left + right) * 0.5

    cv2.line(img, center.astype(int), tip.astype(int), color=(255, 255, 255))
    cv2.circle(img, center=tip.astype(int), radius=5, color=(255, 0, 0), thickness=-1)
    cv2.circle(img, center=left.astype(int), radius=5, color=(0, 0, 255), thickness=-1)
    cv2.circle(img, center=right.astype(int), radius=5, color=(0, 255, 0), thickness=-1)
    cv2.circle(img, center=center.astype(int), radius=5, color=(255, 255, 255), thickness=-1)

    dir = tip - center
    draw_thymio(img, center, np.arctan2(-dir[0], dir[1]))

    cv2.imshow('main', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_obstacle_mask(color_image, source_point):
    threshold = 200
    kernel_size = 50
    img = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernel_size, kernel_size))
    img = cv2.dilate(img, kernel)

    # Flood fill from the robot location, so we only get the contours that
    # are relevant. Note that this requires to know the position of the robot
    # a priori, and might not be suitable if the map has several disconnected
    # regions across which the robot might get kidnapped
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
    assert np.all(img[source_point[1], source_point[0]] == 0), 'Flood fill seed point is not in free space'
    _, img, _, _ = cv2.floodFill(img, mask=mask, seedPoint=source_point, newVal=2)
    _, img = cv2.threshold(img, thresh=1, maxval=1, type=cv2.THRESH_BINARY_INV)

    return img


mouse_x, mouse_y = 0, 0


def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MBUTTONDOWN:
        mouse_x, mouse_y = x, y


def floodfill_background():
    img = cv2.imread('../map.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow('main', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('main', img.shape[1], img.shape[0])
    cv2.setMouseCallback('main', mouse_callback)
    cv2.imshow('main', normalize(img))
    cv2.waitKey(0)

    # Not sure this would actually work well...
    global mouse_x, mouse_y
    seed_point = (mouse_x, mouse_y)
    _, img, mask, _ = cv2.floodFill(img, mask=np.array([], dtype=np.uint8), seedPoint=seed_point, newVal=0, loDiff=2,
                                    upDiff=2, flags=cv2.FLOODFILL_MASK_ONLY)
    # TODO: actually use the mask here
    # _, img = cv2.threshold(img, thresh=1, maxval=1, type=cv2.THRESH_BINARY)
    image_info(mask)
    cv2.imshow('main', np.where(mask[1:-1, 1:-1], img, 0))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # floodfill_background()
    # correct_perspective()
    reconstruct_thymio()
