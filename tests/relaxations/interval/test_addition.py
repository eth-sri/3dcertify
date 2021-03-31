import unittest

import numpy as np
from parameterized import parameterized

from relaxations.interval import Interval
from tests.RelaxationTestCase import RelaxationTestCase, sample_intervals


class TestIntervalAddition(RelaxationTestCase):

    @parameterized.expand([
        [Interval(0, 0), Interval(0, 0), Interval(0, 0)],
        [Interval(1, 1), Interval(1, 1), Interval(2, 2)],
        [Interval(-1, -1), Interval(-2, -2), Interval(-3, -3)],
        [Interval(-2, 5), Interval(1, 2), Interval(-1, 7)],
    ])
    def test_float_intervals(self, a, b, expected):
        self.assertEqual(expected, a + b)
        self.assertEqual(expected, b + a)

    @parameterized.expand(zip(sample_intervals(), sample_intervals()))
    def test_ndarray_intervals(self, a, b):
        self.assertAlmostEqualNumpy(a.lower_bound + b.lower_bound, (a + b).lower_bound)
        self.assertAlmostEqualNumpy(a.lower_bound + b.lower_bound, (b + a).lower_bound)
        self.assertSound(a + b, [a, b], lambda p: p[0] + p[1])

    @parameterized.expand([
        [0, Interval(0, 0), Interval(0, 0)],
        [1, Interval(1, 1), Interval(2, 2)],
        [-1, Interval(-2, -2), Interval(-3, -3)],
        [2, Interval(-2, 5), Interval(0, 7)],
    ])
    def test_with_constants(self, a, b, expected):
        self.assertEqual(expected, a + b)
        self.assertEqual(expected, b + a)

    @parameterized.expand(zip(np.random.uniform(-100, 100, (100, 100)), sample_intervals()))
    def test_with_constant_ndarray(self, a, b):
        self.assertAlmostEqualNumpy(a + b.lower_bound, (a + b).lower_bound)
        self.assertAlmostEqualNumpy(a + b.lower_bound, (b + a).lower_bound)
        self.assertSound(a + b, [b], lambda b: a + b)

    def test_broadcast_addition(self):
        a = Interval(-1, 1)
        b = np.random.uniform(-100, 100, (100,))

        self.assertAlmostEqualNumpy(b - 1, (a + b).lower_bound)
        self.assertAlmostEqualNumpy(b + 1, (a + b).upper_bound)

        self.assertAlmostEqualNumpy(b - 1, (b + a).lower_bound)
        self.assertAlmostEqualNumpy(b + 1, (b + a).upper_bound)


if __name__ == '__main__':
    unittest.main()
