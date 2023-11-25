import timeit

import numpy as np


def segments_intersect_cross(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
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


def segments_intersect_explicit(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    ab = b - a
    cd = d - c
    ac = c - a
    cb_cd = (b[0] - c[:, 0]) * cd[:, 1] - (b[1] - c[:, 1]) * cd[:, 0]
    ca_cd = (a[0] - c[:, 0]) * cd[:, 1] - (a[1] - c[:, 1]) * cd[:, 0]
    ad_ab = (d[:, 0] - a[0]) * ab[1] - (d[:, 1] - a[1]) * ab[0]
    ac_ab = ac[:, 0] * ab[1] - ac[:, 1] * ab[0]

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


def main():
    min_value = 0
    max_value = 400
    for i in range(10000):
        a = np.random.randint(min_value, max_value, 2)
        b = np.random.randint(min_value, max_value, 2)
        c = np.random.randint(min_value, max_value, (200, 2))
        d = np.random.randint(min_value, max_value, (200, 2))
        res1 = segments_intersect_cross(a, b, c, d)
        res2 = segments_intersect_explicit(a, b, c, d)
        assert res1 == res2
    number, t_cross = timeit.Timer(lambda: segments_intersect_cross(a, b, c, d)).autorange()
    t_explicit = timeit.Timer(lambda: segments_intersect_explicit(a, b, c, d)).timeit(number)
    print(f'cross version: {t_cross:.3f} seconds')
    print(f'explicit version: {t_explicit:.3f} seconds ({(t_explicit - t_cross) / t_cross * 100.0:+.2f}%)')


if __name__ == '__main__':
    main()
