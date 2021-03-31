import unittest

import numpy as np
from parameterized import parameterized

import relaxations.interval as iv
from relaxations.interval import Interval
from tests.RelaxationTestCase import RelaxationTestCase, sample_intervals


class TestIntervalSine(RelaxationTestCase):

    @parameterized.expand([
        [Interval(0, 0), Interval(0, 0)],
        [Interval(np.pi / 2, np.pi / 2), Interval(1, 1)],
        [Interval(-np.pi / 2, -np.pi / 2), Interval(-1, -1)],
        [Interval(np.pi * 5 / 2, np.pi * 5 / 2), Interval(1, 1)],
        [Interval(np.pi * 9 / 2, np.pi * 9 / 2), Interval(1, 1)],
        [Interval(0, np.pi / 2), Interval(0, 1)]
    ])
    def test_single_interval(self, a, expected):
        self.assertEqual(expected, iv.sin(a))
        self.assertSound(iv.sin(a), [a], np.sin)

    @parameterized.expand([
        [0, 0],
        [np.pi / 2, 1],
        [-np.pi / 2, -1],
        [np.pi * 5 / 2, 1],
        [np.pi * 9 / 2, 1],
    ])
    def test_single_scalar(self, a, expected):
        self.assertAlmostEqual(expected, iv.sin(a))

    @parameterized.expand([[i] for i in sample_intervals()])
    def test_nd_interval(self, a):
        self.assertSound(iv.sin(a), [a], np.sin)

    @parameterized.expand([[i] for i in np.random.uniform(-100, 100, (100, 100))])
    def test_nd_array(self, a):
        self.assertAlmostEqualNumpy(np.sin(a), iv.sin(a))


class TestIntervalCosine(RelaxationTestCase):

    @parameterized.expand([
        [Interval(0, 0), Interval(1, 1)],
        [Interval(np.pi / 2, np.pi / 2), Interval(0, 0)],
        [Interval(-np.pi / 2, -np.pi / 2), Interval(0, 0)],
        [Interval(np.pi * 5 / 2, np.pi * 5 / 2), Interval(0, 0)],
        [Interval(np.pi * 9 / 2, np.pi * 9 / 2), Interval(0, 0)],
        [Interval(0, np.pi / 2), Interval(0, 1)]
    ])
    def test_single_interval(self, a: Interval, expected: Interval):
        self.assertAlmostEqual(expected.lower_bound, iv.cos(a).lower_bound)
        self.assertAlmostEqual(expected.upper_bound, iv.cos(a).upper_bound)
        self.assertSound(iv.cos(a), [a], np.cos)

    @parameterized.expand([
        [0, 1],
        [np.pi / 2, 0],
        [-np.pi / 2, 0],
        [np.pi * 5 / 2, 0],
        [np.pi * 3, -1],
    ])
    def test_single_scalar(self, a: float, expected: float):
        self.assertAlmostEqual(expected, iv.cos(a))

    @parameterized.expand([[i] for i in sample_intervals()])
    def test_nd_interval(self, a):
        self.assertSound(iv.cos(a), [a], np.cos)

    @parameterized.expand([[i] for i in np.random.uniform(-100, 100, (100, 100))])
    def test_nd_array(self, a):
        self.assertAlmostEqualNumpy(np.cos(a), iv.cos(a))


class TestIntervalSquare(RelaxationTestCase):

    @parameterized.expand([
        [Interval(0, 0), Interval(0, 0)],
        [Interval(1, 1), Interval(1, 1)],
        [Interval(-2, -2), Interval(4, 4)],
        [Interval(2, 3), Interval(4, 9)],
        [Interval(-3, -2), Interval(4, 9)],
        [Interval(-2, 3), Interval(0, 9)]
    ])
    def test_single_interval(self, a: Interval, expected: Interval):
        self.assertAlmostEqual(expected.lower_bound, iv.square(a).lower_bound)
        self.assertAlmostEqual(expected.upper_bound, iv.square(a).upper_bound)
        self.assertSound(iv.square(a), [a], np.square)

    @parameterized.expand([
        [0, 0],
        [1, 1],
        [-1, 1],
        [2, 4],
        [-3, 9],
    ])
    def test_single_scalar(self, a: float, expected: float):
        self.assertAlmostEqual(expected, iv.square(a))

    @parameterized.expand([[i] for i in sample_intervals()])
    def test_nd_interval(self, a):
        self.assertSound(iv.square(a), [a], np.square)

    @parameterized.expand([[i] for i in np.random.uniform(-100, 100, (100, 100))])
    def test_nd_array(self, a):
        self.assertAlmostEqualNumpy(np.square(a), iv.square(a))


if __name__ == '__main__':
    unittest.main()
