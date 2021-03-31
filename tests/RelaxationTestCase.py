import unittest
from typing import List, Union

import numpy as np

from relaxations.interval import Interval
from relaxations.linear_bounds import LinearBounds

ABSOLUTE_TOLERANCE = 1e-11
RELATIVE_TOLERANCE = 1e-9


class RelaxationTestCase(unittest.TestCase):

    def assertSound(self, approximation: Union[Interval, LinearBounds], params: List[Interval], function):
        if isinstance(approximation, Interval):
            self.assertSoundInterval(approximation, params, function)
        elif isinstance(approximation, LinearBounds):
            self.assertSoundBounds(approximation, params, function)
        else:
            assert False, "Unexpected approximation type"

    def assertSoundInterval(self, approximation: Interval, params: List[Interval], function):
        samples = [[np.random.uniform(p.lower_bound, p.upper_bound) for p in params] for _ in range(1000)]
        samples.append([p.lower_bound for p in params])
        samples.append([p.upper_bound for p in params])
        for sampled_params in samples:
            concrete_values = function(sampled_params)
            self.assertAlmostLessThan(concrete_values, approximation.lower_bound)
            self.assertAlmostGreaterThan(concrete_values, approximation.upper_bound)

    def assertSoundBounds(self, approximation: LinearBounds, params: List[Interval], function):
        samples = [[np.random.uniform(p.lower_bound, p.upper_bound) for p in params] for _ in range(1000)]
        samples.append([p.lower_bound for p in params])
        samples.append([p.upper_bound for p in params])
        for sampled_params in samples:
            concrete_value = function(sampled_params)
            lower_bound = approximation.lower_offset
            upper_bound = approximation.upper_offset
            for j, p0 in enumerate(sampled_params):
                lower_bound = lower_bound + approximation.lower_slope[:, :, j] * p0
                upper_bound = upper_bound + approximation.upper_slope[:, :, j] * p0
            self.assertAlmostLessThan(concrete_value, lower_bound)
            self.assertAlmostGreaterThan(concrete_value, upper_bound)

    def assertAlmostLessThan(self, expected: np.ndarray, actual: np.ndarray):
        tolerance = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * np.abs(expected)
        # only enable for debugging as it will significantly increase test runtime
        # self.assertTrue(np.all(actual - tolerance <= expected), f"Max difference: {np.max(expected - actual)}, tolerance: {tolerance}")
        self.assertTrue(np.all(actual - tolerance <= expected))

    def assertAlmostGreaterThan(self, expected: np.ndarray, actual: np.ndarray):
        tolerance = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * np.abs(expected)
        # only enable for debugging as it will significantly increase test runtime
        # self.assertTrue(np.all(actual + tolerance >= expected), f"Max difference: {np.max(expected - actual)}, tolerance: {tolerance}")
        self.assertTrue(np.all(actual + tolerance >= expected))

    def assertAlmostEqualList(self, expected: List, actual: List):
        assert len(expected) == len(actual)
        for a, b in zip(expected, actual):
            if isinstance(a, List):
                self.assertAlmostEqualList(a, b)
            elif isinstance(a, np.ndarray):
                self.assertAlmostEqualNumpy(a, b)
            elif isinstance(a, Interval):
                self.assertAlmostEqualInterval(a, b)
            elif isinstance(a, LinearBounds):
                self.assertAlmostEqualBounds(a, b)
            else:
                assert False, "Invalid list type"

    def assertAlmostEqualNumpy(self, expected: np.ndarray, actual: np.ndarray):
        self.assertEqual(expected.shape, actual.shape)
        np.testing.assert_allclose(actual, expected, rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE)

    def assertAlmostEqualInterval(self, expected: Interval, actual: Interval):
        self.assertAlmostEqualNumpy(expected.lower_bound, actual.lower_bound)
        self.assertAlmostEqualNumpy(expected.upper_bound, actual.upper_bound)

    def assertAlmostEqualBounds(self, expected: LinearBounds, actual: LinearBounds):
        self.assertAlmostEqualNumpy(expected.lower_offset, actual.lower_offset)
        self.assertAlmostEqualNumpy(expected.upper_offset, actual.upper_offset)
        self.assertAlmostEqualNumpy(expected.lower_slope.flatten(), actual.lower_slope.flatten())
        self.assertAlmostEqualNumpy(expected.upper_slope.flatten(), actual.upper_slope.flatten())


def sample_intervals(num_batches=100, size_intervals=100):
    scaling = np.linspace(0, 1, num_batches).reshape((num_batches, 1)).repeat(size_intervals, axis=1)
    center = np.random.uniform(-100 * scaling, 100 * scaling, (num_batches, size_intervals)).squeeze()
    width = np.random.uniform(0, 200 * scaling, (num_batches, size_intervals)).squeeze()
    return [Interval(center[i] - width[i], center[i] + width[i]) for i in range(num_batches)]


def sample_params(num_params=1, num_batches=100):
    scaling = np.linspace(0, 1, num_batches).reshape((num_batches, 1)).repeat(num_params, axis=1)
    center = np.random.uniform(-100 * scaling, 100 * scaling, (num_batches, num_params))
    width = np.random.uniform(0, 4 * scaling, (num_batches, num_params))
    return [[Interval(center[i, j] - width[i, j], center[i, j] + width[i, j]) for j in range(num_params)] for i in range(num_batches)]


def sample_points(num_batches=100, num_points=100):
    scaling = np.linspace(0, 1, num_batches).reshape((num_batches, 1, 1)).repeat(num_points, axis=1).repeat(3, axis=2)
    return np.random.uniform(-100 * scaling, 100 * scaling, (num_batches, num_points, 3))
