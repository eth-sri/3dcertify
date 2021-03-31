import unittest
from functools import partial

import numpy as np
from parameterized import parameterized

from relaxations import taylor
from tests.RelaxationTestCase import RelaxationTestCase, sample_points, sample_params
from transformations.twisting import TwistingZ


class TestTwistingZ(RelaxationTestCase):

    @parameterized.expand(zip(
        sample_points(),
        np.random.uniform(-10, 10, (100, 1))
    ))
    def test_transformation_float(self, x, alpha):
        x_rotated = x[:, 0] * np.cos(alpha * x[:, 2]) - x[:, 1] * np.sin(alpha * x[:, 2])
        y_rotated = x[:, 0] * np.sin(alpha * x[:, 2]) + x[:, 1] * np.cos(alpha * x[:, 2])
        z_rotated = x[:, 2]
        expected = np.stack([x_rotated, y_rotated, z_rotated], axis=1)
        transformation = TwistingZ()
        actual = transformation.transform(x, alpha)
        self.assertAlmostEqualNumpy(expected, actual)

    @parameterized.expand(zip(
        sample_points(),
        sample_params(num_params=1)
    ))
    def test_transformation_interval(self, x, params):
        transformation = TwistingZ()
        actual = transformation.transform(x, params)
        self.assertSound(actual, params, partial(transformation.transform, x))

    @parameterized.expand(zip(
        sample_points(),
        sample_params(num_params=1)
    ))
    def test_transformation_taylor(self, x, params):
        transformation = TwistingZ()
        actual = taylor.encode(transformation, x, params)
        self.assertSound(actual, params, partial(transformation.transform, x))


if __name__ == '__main__':
    unittest.main()
