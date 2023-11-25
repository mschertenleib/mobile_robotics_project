import timeit

from global_map import *
from global_map import _extract_contours


def test_point_polygon(contours, pt):
    distances = [0.0 for _ in range(len(contours))]
    for i in range(len(contours)):
        distances[i] = cv2.pointPolygonTest(contours[i], pt, measureDist=True)
    return distances


def main():
    approx_poly_epsilon = 2
    color_image = cv2.imread('../images/map_divided.png')
    obstacle_mask = get_obstacle_mask(color_image)
    contour_regions = _extract_contours(obstacle_mask, approx_poly_epsilon)
    contour_regions = [contour for region in contour_regions for contour in region]

    number, time_taken = timeit.Timer(lambda: test_point_polygon(contour_regions, (317.0, 405.0))).autorange()
    print(f'Time per call: {time_taken / number} seconds')


if __name__ == '__main__':
    main()
