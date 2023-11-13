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


def intersect_segments(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> bool:
    """
    Checks for intersection between segments a1---a2 and b1---b2.
    a1, a2, b1 and b2 can also be arrays of points, in which case the function returns True if
    any intersection is detected.
    No intersection is reported if some of the end points lie on the other segment without crossing it.
    """
    delta_a = a2 - a1
    delta_b = b2 - b1
    cb1 = np.cross(a2 - b1, delta_b)
    cb2 = np.cross(a1 - b1, delta_b)
    ca1 = np.cross(b2 - a1, delta_a)
    ca2 = np.cross(b1 - a1, delta_a)
    return np.any(
        (((cb1 > 0) & (cb2 < 0)) | ((cb1 < 0) & (cb2 > 0))) & (((ca1 > 0) & (ca2 < 0)) | ((ca1 < 0) & (ca2 > 0))))


def intersect_contours(pt1: np.ndarray, pt2: np.ndarray, contours) -> bool:
    # NOTE: it is very important here that segment_intersects_segment() returns False when two segments only
    # touch but do not cross. If that was not the case (ie. if it returned True), we would have to consider
    # the special case when intersecting an end point with the contour it comes from. Also, due to the kind
    # of vertices we reject prior to doing intersection, we have a second property "for free": any segment
    # between two (non rejected) vertices of a contour, which would intersect said contour, has to cross one
    # its edges. In other words, we can not have a segment that simply connects two opposite vertices of a
    # contour while always staying inside; it would always cross the contour at some point, and hence be
    # detected as intersecting.

    for contour in contours:
        if intersect_segments(pt1, pt2, contour[:-1, 0], contour[1:, 0]):
            return True
        if intersect_segments(pt1, pt2, contour[-1, 0], contour[0, 0]):
            return True
    return False


def extract_static_edges(contours, obstacle_mask):
    edges = []
    line_img = np.zeros(obstacle_mask.shape, dtype=np.uint8)
    for ci in range(len(contours)):
        contour_i = np.squeeze(contours[ci])
        for i in range(len(contour_i)):
            ui = contour_i[i - 1 if i > 0 else len(contour_i) - 1] - contour_i[i]
            vi = contour_i[i + 1 if i < len(contour_i) - 1 else 0] - contour_i[i]
            # Filter out concave vertices
            if np.cross(ui, vi) < 0:
                continue
            for cj in range(ci, len(contours)):
                contour_j = np.squeeze(contours[cj])
                j0 = i + 1 if ci == cj else 0
                for j in range(j0, len(contour_j)):
                    # TODO: we might be able to reduce the amount of work
                    #  here by using some algebra, and leverage numpy's functions
                    uj = contour_j[j - 1 if j > 0 else len(contour_j) - 1] - contour_j[j]
                    vj = contour_j[j + 1 if j < len(contour_j) - 1 else 0] - contour_j[j]

                    # IMPORTANT NOTE: the image space uses a left-hand basis because the y-axis is positive towards
                    # the bottom.

                    # Filter out concave vertices
                    if np.cross(uj, vj) < 0:
                        continue

                    t = contour_j[j] - contour_i[i]
                    n = (-t[1], t[0])
                    ndui = np.dot(n, ui)  # = np.cross(t, ui)
                    ndvi = np.dot(n, vi)  # = np.cross(t, vi)
                    nduj = np.dot(n, uj)  # = np.cross(t, uj)
                    ndvj = np.dot(n, vj)  # = np.cross(t, vj)

                    # Only keep vertices for which the previous and next one both lie on the same
                    # side of the line, or on the line itself.
                    if (ndui < 0 or ndvi < 0) and (ndui > 0 or ndvi > 0):
                        continue
                    if (nduj < 0 or ndvj < 0) and (nduj > 0 or ndvj > 0):
                        continue

                    if True:  # Use bitmap intersection check
                        cv2.line(line_img, contour_i[i], contour_j[j], color=[1])
                        if np.any(line_img & obstacle_mask):
                            line_img = np.zeros(obstacle_mask.shape, dtype=np.uint8)
                            continue
                    else:

                        # The following handles this special case, where the intersection would otherwise not be caught:
                        #            ______
                        #           /######\
                        #    j_____/########\_____i
                        #   /######################\
                        #  /########################\
                        #
                        # Note that this can only happen between vertices of the same contour due to the checks we do
                        # before
                        if ci == cj:
                            if ndui == 0 and np.dot(t, ui) > 0:
                                pim0 = contour_i[i]
                                pim1 = contour_i[i - 1 if i > 0 else len(contour_i) - 1]
                                pim2 = contour_i[i - 2 if i > 1 else len(contour_i) - 1]
                                if np.cross(pim2 - pim1, pim0 - pim1) < 0:
                                    continue
                            if ndvi == 0 and np.dot(t, vi) > 0:
                                pip0 = contour_i[i]
                                pip1 = contour_i[i + 1 if i < len(contour_i) - 1 else 0]
                                pip2 = contour_i[i + 2 if i < len(contour_i) - 2 else 0]
                                if np.cross(pip0 - pip1, pip2 - pip1) < 0:
                                    continue
                            if nduj == 0 and np.dot(t, uj) < 0:
                                pjm0 = contour_j[j]
                                pjm1 = contour_j[j - 1 if j > 0 else len(contour_j) - 1]
                                pjm2 = contour_j[j - 2 if j > 1 else len(contour_j) - 1]
                                if np.cross(pjm2 - pjm1, pjm0 - pjm1) < 0:
                                    continue
                            if ndvj == 0 and np.dot(t, vj) < 0:
                                pjp0 = contour_j[j]
                                pjp1 = contour_j[j + 1 if j < len(contour_j) - 1 else 0]
                                pjp2 = contour_j[j + 2 if j < len(contour_j) - 2 else 0]
                                if np.cross(pjp0 - pjp1, pjp2 - pjp1) < 0:
                                    continue

                        # Discard edges that intersect contours
                        if intersect_contours(contour_i[i], contour_j[j], contours):
                            continue

                    edges.append([contour_i[i].tolist(), contour_j[j].tolist()])

    return edges


def extract_dynamic_edges(contours, obstacle_mask, point):
    # FIXME: We should try to avoid duplication anyway.
    line_img = np.zeros(obstacle_mask.shape, dtype=np.uint8)
    edges = []
    for cj in range(len(contours)):
        contour_j = np.squeeze(contours[cj])
        for j in range(len(contour_j)):
            uj = contour_j[j - 1 if j > 0 else len(contour_j) - 1] - contour_j[j]
            vj = contour_j[j + 1 if j < len(contour_j) - 1 else 0] - contour_j[j]

            if np.cross(uj, vj) < 0:
                continue

            t = contour_j[j] - point
            n = (-t[1], t[0])
            nduj = np.dot(n, uj)
            ndvj = np.dot(n, vj)

            if (nduj < 0 or ndvj < 0) and (nduj > 0 or ndvj > 0):
                continue

            cv2.line(line_img, point, contour_j[j], color=[1])
            if np.any(line_img & obstacle_mask):
                line_img = np.zeros(obstacle_mask.shape, dtype=np.uint8)
                continue

            edges.append([point.tolist(), contour_j[j].tolist()])

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
    start_point = (200, 100)
    target_point = (120, 730)
    original_img = cv2.imread('map.png')
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
    assert img[start_point[::-1]] == 0, 'Flood fill seed point is not in free space'
    _, img, _, _ = cv2.floodFill(img, mask=mask,
                                 seedPoint=start_point, newVal=2)
    _, img = cv2.threshold(img, thresh=1, maxval=1, type=cv2.THRESH_BINARY_INV)

    # NOTE: using RETR_EXTERNAL means we would only get the outer contours,
    # which is basically equivalent to floodfilling the binary image from the
    # outside prior to calling findContours. However, this assumes our region
    # of interest is "outside", and would not work if we are "walled in",
    # which is actually very likely
    contours, hierarchy = cv2.findContours(img, mode=cv2.RETR_LIST,
                                           method=cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.approxPolyDP(contour, epsilon=approx_poly_epsilon, closed=True) for contour in contours]

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
    cv2.drawContours(img, contours, contourIdx=-1, color=(192, 64, 64))

    obstacle_mask = np.zeros(original_img.shape[0:2], dtype=np.uint8)
    cv2.drawContours(obstacle_mask, contours, contourIdx=-1, color=[1])
    mask = np.zeros((obstacle_mask.shape[0] + 2, obstacle_mask.shape[1] + 2), dtype=np.uint8)
    _, obstacle_mask, _, _ = cv2.floodFill(obstacle_mask, mask=mask, seedPoint=start_point, newVal=[1])
    assert np.min(obstacle_mask) == 0
    assert np.max(obstacle_mask) == 1
    obstacle_mask = 1 - obstacle_mask

    edges = extract_static_edges(contours, obstacle_mask)
    print(f'Number of static edges: {len(edges)}')
    start_edges = extract_dynamic_edges(contours, obstacle_mask, np.array(start_point))
    target_edges = extract_dynamic_edges(contours, obstacle_mask, np.array(target_point))
    print(f'Number of dynamic edges: {len(start_edges) + len(target_edges)}')

    for edge in edges:
        cv2.line(img, edge[0], edge[1], color=(0, 0, 0))
    for edge in start_edges:
        cv2.line(img, edge[0], edge[1], color=(64, 192, 64))
    for edge in target_edges:
        cv2.line(img, edge[0], edge[1], color=(64, 64, 192))
    cv2.circle(img, start_point, color=(64, 192, 64), radius=6, thickness=-1)
    cv2.circle(img, target_point, color=(64, 64, 192), radius=6, thickness=-1)
    img = normalize(img)
    cv2.namedWindow('main', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('main', img.shape[1], img.shape[0])
    cv2.imshow('main', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
