import timeit

import numpy as np


def intersect_segments_cross(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> bool:
    delta_a = a2 - a1
    delta_b = b2 - b1
    cb1 = np.cross(a2 - b1, delta_b)
    cb2 = np.cross(a1 - b1, delta_b)
    ca1 = np.cross(b2 - a1, delta_a)
    ca2 = np.cross(b1 - a1, delta_a)
    return np.any(
        (((cb1 >= 0) & (cb2 <= 0)) | ((cb1 <= 0) & (cb2 >= 0))) & (
                ((ca1 >= 0) & (ca2 <= 0)) | ((ca1 <= 0) & (ca2 >= 0))))


def intersect_segments_dot(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> bool:
    delta_a = np.array([a1[1] - a2[1], a2[0] - a1[0]])
    delta_b = np.array([b1[:, 1] - b2[:, 1], b2[:, 0] - b1[:, 0]])
    cb1 = (a2[0] - b1[:, 0]) * delta_b[0, :] + (a2[1] - b1[:, 1]) * delta_b[1, :]
    cb2 = (a1[0] - b1[:, 0]) * delta_b[0, :] + (a1[1] - b1[:, 1]) * delta_b[1, :]
    ca1 = (b2[:, 0] - a1[0]) * delta_a[0] + (b2[:, 1] - a1[1]) * delta_a[1]
    ca2 = (b1[:, 0] - a1[0]) * delta_a[0] + (b1[:, 1] - a1[1]) * delta_a[1]
    return np.any(
        (((cb1 >= 0) & (cb2 <= 0)) | ((cb1 <= 0) & (cb2 >= 0))) & (
                ((ca1 >= 0) & (ca2 <= 0)) | ((ca1 <= 0) & (ca2 >= 0))))


if __name__ == '__main__':
    min_value = 0
    max_value = 400
    nt = 0
    nf = 0
    for i in range(10000):
        a1 = np.random.randint(min_value, max_value, 2)
        a2 = np.random.randint(min_value, max_value, 2)
        b1 = np.random.randint(min_value, max_value, (200, 2))
        b2 = np.random.randint(min_value, max_value, (200, 2))
        res1 = intersect_segments_cross(a1, a2, b1, b2)
        res2 = intersect_segments_dot(a1, a2, b1, b2)
        assert res1 == res2
    repeats = 10000
    t_cross = timeit.Timer(lambda: intersect_segments_cross(a1, a2, b1, b2)).timeit(repeats)
    t_dot = timeit.Timer(lambda: intersect_segments_dot(a1, a2, b1, b2)).timeit(repeats)
    print(f'cross version: {t_cross:.3f} seconds')
    print(f'dot version: {t_dot:.3f} seconds ({(t_dot - t_cross) / t_cross * 100.0:+.2f}%)')
