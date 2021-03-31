import unittest

import numpy as np
from parameterized import parameterized

from relaxations.interval import Interval
from tests.RelaxationTestCase import RelaxationTestCase, sample_intervals


class TestIntervalMultiplication(RelaxationTestCase):

    @parameterized.expand([
        [Interval(0, 0), Interval(0, 0), Interval(0, 0)],
        [Interval(1, 1), Interval(1, 1), Interval(1, 1)],
        [Interval(-1, -1), Interval(-2, -2), Interval(2, 2)],
        [Interval(-2, 2), Interval(-2, 2), Interval(-4, 4)],
        [Interval(-2, 5), Interval(1, 2), Interval(-4, 10)],
    ])
    def test_float_intervals(self, a, b, expected):
        self.assertEqual(expected, a * b)
        self.assertEqual(expected, b * a)

    @parameterized.expand(zip(sample_intervals(), sample_intervals()))
    def test_ndarray_intervals(self, a, b):
        self.assertSound(a * b, [a, b], lambda p: p[0] * p[1])
        self.assertSound(b * a, [a, b], lambda p: p[0] * p[1])

    @parameterized.expand([
        [0, Interval(0, 0), Interval(0, 0)],
        [1, Interval(1, 1), Interval(1, 1)],
        [-1, Interval(-2, -2), Interval(2, 2)],
        [2, Interval(-2, 5), Interval(-4, 10)],
    ])
    def test_with_constants(self, a, b, expected):
        self.assertEqual(expected, a * b)
        self.assertEqual(expected, b * a)

    @parameterized.expand(zip(np.random.uniform(-100, 100, (100, 100)), sample_intervals()))
    def test_with_constant_ndarray(self, a, b):
        self.assertSound(a * b, [b], lambda b: a * b)
        self.assertSound(b * a, [b], lambda b: a * b)

    def test_broadcast_addition(self):
        a = Interval(-1., 1.)
        b = np.random.uniform(-100, 100, (100,))

        self.assertAlmostEqualNumpy(np.minimum(-b, b), (a * b).lower_bound)
        self.assertAlmostEqualNumpy(np.maximum(-b, b), (a * b).upper_bound)

        self.assertAlmostEqualNumpy(np.minimum(-b, b), (b * a).lower_bound)
        self.assertAlmostEqualNumpy(np.maximum(-b, b), (b * a).upper_bound)


if __name__ == '__main__':
    unittest.main()
