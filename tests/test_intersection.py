import numpy as np

from global_map import segments_intersect


class TestCase:
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
        TestCase((8, -4), (10, 2), False, 'CD to the left'),
        TestCase((8, 0), (12, -3), False, 'CD to the left'),
        TestCase((0, 0), (6, 4), False, 'CD to the right'),
        TestCase((4, 2), (0, 5), False, 'CD to the right'),
        # Intersecting
        TestCase((2, 2), (8, 0), True, 'crossing'),
        # Colinear not intersecting
        TestCase((-1, -7), (2, 2), False, 'C or D colinear with AB, after A'),
        TestCase((-1, -7), (0, -9), False, 'C or D colinear with AB, after A'),
        TestCase((11, 9), (4, 4), False, 'C or D colinear with AB, after B'),
        TestCase((11, 9), (14, 8), False, 'C or D colinear with AB, after B'),
        TestCase((-1, -7), (-4, -11), False, 'CD colinear with AB, after A'),
        TestCase((11, 9), (14, 13), False, 'CD colinear with AB, after B'),
        # Colinear intersecting
        TestCase((5, 1), (6.5, 3), True, 'CD colinear with AB, CD inside AB'),
        TestCase((-1, -7), (11, 9), True, 'CD colinear with AB, AB inside CD'),
        TestCase((5, 1), (11, 9), True, 'CD colinear with AB, half superposition, after B'),
        TestCase((5, 1), (-1, -7), True, 'CD colinear with AB, half superposition, after A'),
        # Contact: vertex AB - vertex CD
        TestCase((2, -3), (-1, -7), False, 'CD colinear with AB, C or D on A'),
        TestCase((8, 5), (11, 9), False, 'CD colinear with AB, C or D on B'),
        TestCase((2, -3), (-4, -1), False, 'CD to the left, C or D on A'),
        TestCase((2, -3), (4, -6), False, 'CD to the right, C or D on A'),
        TestCase((8, 5), (4, 6), False, 'CD to the left, C or D on B'),
        TestCase((8, 5), (12, 4), False, 'CD to the right, C or D on B'),
        # Contact: vertex AB - edge CD
        TestCase((-1, 1), (5, -7), False, 'A on CD'),
        TestCase((5, 9), (11, 1), False, 'B on CD'),
        # Contact: edge AB - vertex CD
        TestCase((5, 1), (1, 3), True, 'CD to the left, C or D on AB'),
        TestCase((5, 1), (10, -2), True, 'CD to the right, C or D on AB')]

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
