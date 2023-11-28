from global_map import *

import cv2


def draw_distance_to_contours(img_width: int, img_height: int, regions: list[list[cv2.typing.MatLike]]):
    dist = np.zeros((img_height, img_width), dtype=np.float32) - np.inf
    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            for region_contours in regions:
                distance, _ = distance_to_contours((j, i), region_contours)
                dist[i, j] = max(dist[i, j], distance)

    min_dist, max_dist, min_dist_pt, max_dist_pt = cv2.minMaxLoc(dist)
    min_dist = abs(min_dist)
    max_dist = abs(max_dist)

    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    dist_mask = dist >= 0
    img[dist_mask, 1] = dist[dist_mask] / max_dist * 255
    img[~dist_mask, 2] = -dist[~dist_mask] / min_dist * 255

    flattened_contours = [contour for region in regions for contour in region]
    cv2.drawContours(img, flattened_contours, contourIdx=-1, color=(255, 255, 255))

    return img


def main():
    approx_poly_epsilon = 2
    color_image = cv2.imread('../images/map_divided.png')
    obstacle_mask = get_obstacle_mask(color_image)
    regions = extract_contours(obstacle_mask, approx_poly_epsilon)
    img = draw_distance_to_contours(color_image.shape[1], color_image.shape[0], regions)

    cv2.namedWindow('main', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('main', img.shape[1], img.shape[0])
    cv2.imshow('main', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
