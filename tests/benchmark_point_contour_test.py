import timeit

import numpy as np

from global_map import *


def main():
    approx_poly_epsilon = 2
    color_image = cv2.imread('../images/map_divided.png')
    obstacle_mask = get_obstacle_mask(color_image)
    regions = extract_contours(obstacle_mask, approx_poly_epsilon)
    image_size = np.array(color_image.shape[1::-1], dtype=float)
    number, time_taken = timeit.Timer(
        lambda: distance_to_contours(np.random.random(2) * image_size, regions[3])).autorange()
    print(f'Time taken: {time_taken} seconds, {number} repeats (time per call: {time_taken / number} seconds)')


if __name__ == '__main__':
    main()
