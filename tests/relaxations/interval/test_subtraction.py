import unittest

import numpy as np
from parameterized import parameterized

from relaxations.interval import Interval
from tests.RelaxationTestCase import RelaxationTestCase, sample_intervals


class TestIntervalSubtraction(RelaxationTestCase):

    @parameterized.expand([
        [Interval(0, 0), Interval(0, 0), Interval(0, 0)],
        [Interval(1, 1), Interval(1, 1), Interval(0, 0)],
        [Interval(1, 1), Interval(-2, -2), Interval(3, 3)],
        [Interval(-2, 5), Interval(1, 2), Interval(-4, 4)],
    ])
    def test_float_intervals(self, a, b, expected):
        self.assertEqual(expected, a - b)

    @parameterized.expand(zip(sample_intervals(), sample_intervals()))
    def test_ndarray_intervals(self, a, b):
        self.assertAlmostEqualNumpy(a.lower_bound - b.upper_bound, (a - b).lower_bound)
        self.assertAlmostEqualNumpy(a.upper_bound - b.lower_bound, (a - b).upper_bound)
        self.assertSound(a - b, [a, b], lambda p: p[0] - p[1])

    @parameterized.expand([
        [0, Interval(0, 0), Interval(0, 0)],
        [1, Interval(1, 1), Interval(0, 0)],
        [-1, Interval(-2, -2), Interval(1, 1)],
        [2, Interval(-2, 5), Interval(-3, 4)],
    ])
    def test_with_constants(self, a, b, expected):
        self.assertEqual(expected, a - b)

    @parameterized.expand(zip(np.random.uniform(-100, 100, (100, 100)), sample_intervals()))
    def test_with_constant_ndarray(self, a, b):
        self.assertAlmostEqualNumpy(a - b.upper_bound, (a - b).lower_bound)
        self.assertAlmostEqualNumpy(a - b.lower_bound, (a - b).upper_bound)
        self.assertSound(a - b, [b], lambda b: a - b)

        self.assertAlmostEqualNumpy(b.lower_bound - a, (b - a).lower_bound)
        self.assertAlmostEqualNumpy(b.upper_bound - a, (b - a).upper_bound)
        self.assertSound(b - a, [b], lambda b: b - a)

    def test_broadcast(self):
        a = Interval(-1, 1)
        b = np.random.uniform(-100, 100, (100,))

        self.assertAlmostEqualNumpy(-b - 1, (a - b).lower_bound)
        self.assertAlmostEqualNumpy(-b + 1, (a - b).upper_bound)

        self.assertAlmostEqualNumpy(b - 1, (b - a).lower_bound)
        self.assertAlmostEqualNumpy(b + 1, (b - a).upper_bound)

    @parameterized.expand([
        [Interval(0, 0), Interval(0, 0)],
        [Interval(1, 1), Interval(-1, -1)],
        [Interval(-2, 2), Interval(-2, 2)],
        [Interval(-2, 5), Interval(-5, 2)],
    ])
    def test_interval_inversion(self, a, expected):
        self.assertEqual(expected, -a)


if __name__ == '__main__':
    unittest.main()
