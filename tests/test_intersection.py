import numpy as np


def segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    ab = b - a
    cd = d - c
    ac = c - a
    cb_cd = (b[0] - c[:, 0]) * cd[:, 1] - (b[1] - c[:, 1]) * cd[:, 0]
    ca_cd = (a[0] - c[:, 0]) * cd[:, 1] - (a[1] - c[:, 1]) * cd[:, 0]
    ad_ab = (d[:, 0] - a[0]) * ab[1] - (d[:, 1] - a[1]) * ab[0]
    ac_ab = ac[:, 0] * ab[1] - ac[:, 1] * ab[0]

    # Detects intersections between non-colinear segments
    if np.any(
            (((cb_cd > 0) & (ca_cd < 0)) | ((cb_cd < 0) & (ca_cd > 0))) & (
                    ((ad_ab >= 0) & (ac_ab <= 0)) | ((ad_ab <= 0) & (ac_ab >= 0)))):
        return True

    colinear_mask = (cb_cd == 0) & (ca_cd == 0) & (ad_ab == 0) & (ac_ab == 0)
    ac_dot_ab = ac[:, 0] * ab[0] + ac[:, 1] * ab[1]
    ab_dot_cd = ab[0] * cd[:, 0] + ab[1] * cd[:, 1]
    ab_dot_ab = ab[0] * ab[0] + ab[1] * ab[1]
    mask_dir_same = (ab_dot_cd > 0) & (ac_dot_ab + ab_dot_cd > 0) & (ac_dot_ab - ab_dot_ab < 0)
    mask_dir_opposite = (ab_dot_cd < 0) & (ac_dot_ab > 0) & (ac_dot_ab + ab_dot_cd - ab_dot_ab < 0)
    return np.any(colinear_mask & (mask_dir_same | mask_dir_opposite))


class Test_case:
    def __init__(self, c: tuple, d: tuple, expected: bool, info: str):
        self.c = c
        self.d = d
        self.expected = expected
        self.info = info


def main():
    a = np.array([2, -3])
    b = np.array([8, 5])

    test_cases = [
        # Not intersecting
        Test_case((8, -4), (10, 2), False, 'CD to the left'),
        Test_case((8, 0), (12, -3), False, 'CD to the left'),
        Test_case((0, 0), (6, 4), False, 'CD to the right'),
        Test_case((4, 2), (0, 5), False, 'CD to the right'),
        # Intersecting
        Test_case((2, 2), (8, 0), True, 'crossing'),
        # Colinear not intersecting
        Test_case((-1, -7), (2, 2), False, 'C or D colinear with AB, after A'),
        Test_case((-1, -7), (0, -9), False, 'C or D colinear with AB, after A'),
        Test_case((11, 9), (4, 4), False, 'C or D colinear with AB, after B'),
        Test_case((11, 9), (14, 8), False, 'C or D colinear with AB, after B'),
        Test_case((-1, -7), (-4, -11), False, 'CD colinear with AB, after A'),
        Test_case((11, 9), (14, 13), False, 'CD colinear with AB, after B'),
        # Colinear intersecting
        Test_case((5, 1), (6.5, 3), True, 'CD colinear with AB, CD inside AB'),
        Test_case((-1, -7), (11, 9), True, 'CD colinear with AB, AB inside CD'),
        Test_case((5, 1), (11, 9), True, 'CD colinear with AB, half superposition, after B'),
        Test_case((5, 1), (-1, -7), True, 'CD colinear with AB, half superposition, after A'),
        # Contact: vertex AB - vertex CD
        Test_case((2, -3), (-1, -7), False, 'CD colinear with AB, C or D on A'),
        Test_case((8, 5), (11, 9), False, 'CD colinear with AB, C or D on B'),
        Test_case((2, -3), (-4, -1), False, 'CD to the left, C or D on A'),
        Test_case((2, -3), (4, -6), False, 'CD to the right, C or D on A'),
        Test_case((8, 5), (4, 6), False, 'CD to the left, C or D on B'),
        Test_case((8, 5), (12, 4), False, 'CD to the right, C or D on B'),
        # Contact: vertex AB - edge CD
        Test_case((-1, 1), (5, -7), False, 'A on CD'),
        Test_case((5, 9), (11, 1), False, 'B on CD'),
        # Contact: edge AB - vertex CD
        Test_case((5, 1), (1, 3), True, 'CD to the left, C or D on AB'),
        Test_case((5, 1), (10, -2), True, 'CD to the right, C or D on AB')]

    all_passed = True
    for test_case in test_cases:
        c = np.array([test_case.c])
        d = np.array([test_case.d])
        for c, d in ((c, d), (d, c)):
            test = segments_intersect(a, b, c, d)
            if test != test_case.expected:
                all_passed = False
                print(
                    f'Test failed: {test_case.info} (with C = {c}, D = {d}). Expected {test_case.expected}, got {test}')

    if all_passed:
        print('All tests succeeded')


if __name__ == '__main__':
    main()
