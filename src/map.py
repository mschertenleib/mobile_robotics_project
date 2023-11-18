import cv2
import numpy as np


def image_info(img: np.ndarray):
    print(
        f'dtype: {img.dtype}, shape: {img.shape}, min: {img.min()}, max: {img.max()}')


def normalize(img: np.ndarray):
    range_values = img.max() - img.min()
    if range_values == 0:
        return img.astype(np.float32)
    else:
        return (img.astype(np.float32) - img.min()) / range_values


def segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    """
    Checks for an intersection between the half-open segment ]a---b] and closed segments [c[i]---d[i]]
    """

    ab = b - a
    cd = d - c
    ac = c - a
    cb_cd = np.cross(b - c, cd)
    ca_cd = np.cross(a - c, cd)
    ad_ab = np.cross(d - a, ab)
    ac_ab = np.cross(ac, ab)

    # Detects intersections between non-colinear segments
    if np.any(
            (((cb_cd >= 0) & (ca_cd < 0)) | ((cb_cd <= 0) & (ca_cd > 0))) & (
                    ((ad_ab > 0) & (ac_ab < 0)) | ((ad_ab < 0) & (ac_ab > 0)))):
        return True

    colinear_mask = (cb_cd == 0) & (ca_cd == 0) & (ad_ab == 0) & (ac_ab == 0)
    ac_dot_ab = ac[:, 0] * ab[0] + ac[:, 1] * ab[1]
    ab_dot_cd = ab[0] * cd[:, 0] + ab[1] * cd[:, 1]
    ab_dot_ab = ab[0] * ab[0] + ab[1] * ab[1]
    mask_dir_same = (ab_dot_cd > 0) & (ac_dot_ab + ab_dot_cd > 0) & (ac_dot_ab - ab_dot_ab <= 0)
    mask_dir_opposite = (ab_dot_cd < 0) & (ac_dot_ab > 0) & (ac_dot_ab + ab_dot_cd - ab_dot_ab <= 0)
    return np.any(colinear_mask & (mask_dir_same | mask_dir_opposite))


def segment_intersects_contours(pt1: np.ndarray, pt2: np.ndarray, contours) -> bool:
    """
    Checks for an intersection between the half-open segment ]pt1---pt2] and all contours
    """
    for contour in contours:
        if segments_intersect(pt1, pt2, contour[:-1], contour[1:]):
            return True
        if segments_intersect(pt1, pt2, np.array([contour[-1]]), np.array([contour[0]])):
            return True
    return False


def extract_convex_vertices(contours):
    vertices = []
    to_prev = []
    to_next = []
    for contour in contours:
        for i in range(len(contour)):
            to_prev_i = contour[i - 1 if i > 0 else len(contour) - 1] - contour[i]
            to_next_i = contour[i + 1 if i < len(contour) - 1 else 0] - contour[i]
            if np.cross(to_prev_i, to_next_i) >= 0:
                vertices.append(contour[i])
                to_prev.append(to_prev_i)
                to_next.append(to_prev_i)
    return vertices, to_prev, to_next


def extract_static_edges_old(contours):
    edges = []
    for ic in range(len(contours)):
        for iv in range(len(contours[ic])):
            # Filter out concave vertices
            to_prev_i = contours[ic][iv - 1 if iv > 0 else len(contours[ic]) - 1] - contours[ic][iv]
            to_next_i = contours[ic][iv + 1 if iv < len(contours[ic]) - 1 else 0] - contours[ic][iv]
            if np.cross(to_prev_i, to_next_i) < 0:
                continue

            edges += extract_edges_old(contours, ic, iv, to_prev_i, to_next_i)

    return edges


def extract_edges_old(contours, ic, iv, to_prev_i, to_next_i):
    # IMPORTANT NOTE: the image space uses a left-hand basis because the y-axis is positive towards
    # the bottom.

    edges = []

    jc = ic
    for jv in range(iv + 1, len(contours[jc])):
        # Discard concave vertices
        to_prev_j = contours[jc][jv - 1 if jv > 0 else len(contours[jc]) - 1] - contours[jc][jv]
        to_next_j = contours[jc][jv + 1 if jv < len(contours[jc]) - 1 else 0] - contours[jc][jv]
        # FIXME: we are checking for convexity over and over again for the same vertex. It is probably much better to
        #  precompute the to_prev and to_next for each vertex, then pass that to a function that does the core loop
        #  and checks. It would also probably make the whole system clearer and more modular
        if np.cross(to_prev_j, to_next_j) < 0:
            continue

        # Only keep vertices for which the previous and next one both lie on the same
        # side of the line, or on the line itself.
        t = contours[jc][jv] - contours[ic][iv]
        sin_prev_i = np.cross(t, to_prev_i)
        sin_next_i = np.cross(t, to_next_i)
        if (sin_prev_i < 0 or sin_next_i < 0) and (sin_prev_i > 0 or sin_next_i > 0):
            continue
        sin_prev_j = np.cross(t, to_prev_j)
        sin_next_j = np.cross(t, to_next_j)
        if (sin_prev_j < 0 or sin_next_j < 0) and (sin_prev_j > 0 or sin_next_j > 0):
            continue

        # Discard edges that intersect contours
        is_neighbor = (jv == iv + 1 or (iv == 0 and jv == len(contours[jc]) - 1))
        if not is_neighbor and segment_intersects_contours(contours[ic][iv], contours[jc][jv], contours):
            continue

        edges.append([ic, iv, jc, jv])

    for jc in range(ic + 1, len(contours)):
        for jv in range(0, len(contours[jc])):
            # Discard concave vertices
            to_prev_j = contours[jc][jv - 1 if jv > 0 else len(contours[jc]) - 1] - contours[jc][jv]
            to_next_j = contours[jc][jv + 1 if jv < len(contours[jc]) - 1 else 0] - contours[jc][jv]
            if np.cross(to_prev_j, to_next_j) < 0:
                continue

            # Only keep vertices for which the previous and next one both lie on the same
            # side of the line, or on the line itself.
            t = contours[jc][jv] - contours[ic][iv]
            sin_prev_i = np.cross(t, to_prev_i)
            sin_next_i = np.cross(t, to_next_i)
            if (sin_prev_i < 0 or sin_next_i < 0) and (sin_prev_i > 0 or sin_next_i > 0):
                continue
            sin_prev_j = np.cross(t, to_prev_j)
            sin_next_j = np.cross(t, to_next_j)
            if (sin_prev_j < 0 or sin_next_j < 0) and (sin_prev_j > 0 or sin_next_j > 0):
                continue

            # Discard edges that intersect contours
            if segment_intersects_contours(contours[ic][iv], contours[jc][jv], contours):
                continue

            edges.append([ic, iv, jc, jv])

    return edges


def extract_dynamic_edges_old(contours, point):
    edges = []
    for jc in range(0, len(contours)):
        for jv in range(0, len(contours[jc])):
            # Discard concave vertices
            to_prev_j = contours[jc][jv - 1 if jv > 0 else len(contours[jc]) - 1] - contours[jc][jv]
            to_next_j = contours[jc][jv + 1 if jv < len(contours[jc]) - 1 else 0] - contours[jc][jv]
            if np.cross(to_prev_j, to_next_j) < 0:
                continue

            # Only keep vertices for which the previous and next one both lie on the same
            # side of the line, or on the line itself.
            t = contours[jc][jv] - point
            sin_prev_j = np.cross(t, to_prev_j)
            sin_next_j = np.cross(t, to_next_j)
            if (sin_prev_j < 0 or sin_next_j < 0) and (sin_prev_j > 0 or sin_next_j > 0):
                continue

            # Discard edges that intersect contours
            if segment_intersects_contours(point, contours[jc][jv], contours):
                continue

            edges.append([point, contours[jc][jv].tolist()])

    return edges


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
            cv2.circle(img, contours[c][i][0], color=(brightness, brightness, brightness), radius=5, thickness=-1)


def main():
    threshold = 200
    kernel_size = 50
    # Note: the minimum distance to any obstacle is 'kernel_size - approx_poly_epsilon'
    approx_poly_epsilon = 2
    source_point = (200, 100)
    target_point = (120, 730)
    original_img = cv2.imread('../map.png')
    img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernel_size, kernel_size))
    img = cv2.dilate(img, kernel)

    # Flood fill from the robot location, so we only get the contours that
    # are relevant. Note that this requires to know the position of the robot
    # a priori, and might not be suitable if the map has several disconnected
    # regions across which the robot might get kidnapped
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
    assert img[source_point[::-1]] == 0, 'Flood fill seed point is not in free space'
    _, img, _, _ = cv2.floodFill(img, mask=mask,
                                 seedPoint=source_point, newVal=2)
    _, img = cv2.threshold(img, thresh=1, maxval=1, type=cv2.THRESH_BINARY_INV)

    # NOTE: using RETR_EXTERNAL means we would only get the outer contours,
    # which is basically equivalent to floodfilling the binary image from the
    # outside prior to calling findContours. However, this assumes our region
    # of interest is "outside", and would not work if we are "walled in",
    # which is actually very likely
    contours, hierarchy = cv2.findContours(img, mode=cv2.RETR_LIST,
                                           method=cv2.CHAIN_APPROX_SIMPLE)
    contours = [np.squeeze(cv2.approxPolyDP(contour, epsilon=approx_poly_epsilon, closed=True)) for contour in contours]

    # NOTE: orientation is positive for a clockwise contour, which is the opposite of the mathematical standard
    # (right-hand rule). Note however that the outer contour of a shape is always
    # counter-clockwise (hence orientation is negative), and the contour of a hole is always clockwise (hence
    # orientation is positive).
    orientations = [np.sign(cv2.contourArea(contour, oriented=True)) for contour in contours]

    # If we know that 'contours' only contains those in the region where the
    # robot is located, we can use them directly for computing the visibility
    # graph. Else, we must take into account the hierarchy.

    walkable = np.zeros(original_img.shape, dtype=np.uint8)
    walkable[:] = (192, 64, 64)
    cv2.drawContours(walkable, contours, contourIdx=-1, color=(255, 255, 255), thickness=-1)
    img = cv2.addWeighted(original_img, 0.75, walkable, 0.25, 0.0)
    cv2.drawContours(img, contours, contourIdx=-1, color=(64, 64, 192))

    vertices, to_prev, to_next = extract_convex_vertices(contours)
    print(f'{len(vertices) = }')
    print(f'{len(to_prev) = }')
    print(f'{len(to_next) = }')
    exit()

    edges = extract_static_edges_old(contours)
    print(f'Number of static edges: {len(edges)}')
    source_edges = extract_dynamic_edges_old(contours, np.array(source_point))
    target_edges = extract_dynamic_edges_old(contours, np.array(target_point))
    print(f'Number of dynamic edges: {len(source_edges) + len(target_edges)}')

    for ci, i, cj, j in edges:
        cv2.line(img, contours[ci][i], contours[cj][j], color=(0, 0, 0))
    for edge in source_edges:
        cv2.line(img, edge[0], edge[1], color=(64, 192, 64))
    for edge in target_edges:
        cv2.line(img, edge[0], edge[1], color=(64, 64, 192))
    cv2.circle(img, source_point, color=(64, 192, 64), radius=6, thickness=-1)
    cv2.circle(img, target_point, color=(64, 64, 192), radius=6, thickness=-1)
    # draw_contour_orientations(img, contours, orientations)
    cv2.namedWindow('main', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('main', img.shape[1], img.shape[0])
    cv2.imshow('main', normalize(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


mouse_x, mouse_y = 0, 0


def on_click(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_x, mouse_y = x, y


def floodfill_background():
    img = cv2.imread('../map.png')
    img = cv2.cv2tColor(img, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow('main', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('main', img.shape[1], img.shape[0])
    cv2.setMouseCallback('main', on_click)
    cv2.imshow('main', normalize(img))
    cv2.waitKey(0)

    # Not sure this would actually work well...
    global mouse_x, mouse_y
    seed_point = (mouse_x, mouse_y)
    _, img, mask, _ = cv2.floodFill(img, mask=np.array([], dtype=np.uint8), seedPoint=seed_point, newVal=0, loDiff=2,
                                    upDiff=2, flags=cv2.FLOODFILL_MASK_ONLY)
    # TOOD: actually use the mask here
    # _, img = cv2.threshold(img, thresh=1, maxval=1, type=cv2.THRESH_BINARY)
    image_info(mask)
    cv2.imshow('main', np.where(mask[1:-1, 1:-1], img, 0))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
    # floodfill_background()

    """
    a1 = np.array([3.7, -1.9])
    a2 = np.array([10.67, -5.09])
    b1 = np.array([3.7, -1.9])
    b2 = np.array([3.14, 7.5])
    print(intersect_segments_closed(a1, a2, b1, b2))
    """
