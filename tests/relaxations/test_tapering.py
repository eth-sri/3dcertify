import unittest
from functools import partial

import numpy as np
from parameterized import parameterized

from relaxations import taylor
from tests.RelaxationTestCase import RelaxationTestCase, sample_points, sample_params
from transformations.tapering import TaperingZ


class TestTaperingZ(RelaxationTestCase):

    @parameterized.expand(zip(
        sample_points(),
        np.random.uniform(-10, 10, (100, 2))
    ))
    def test_transformation_float(self, x, params):
        a, b = params
        transformed_x = (0.5 * np.square(a) * x[:, 2] + b * x[:, 2] + 1) * x[:, 0]
        transformed_y = (0.5 * np.square(a) * x[:, 2] + b * x[:, 2] + 1) * x[:, 1]
        transformed_z = x[:, 2]
        expected = np.stack([transformed_x, transformed_y, transformed_z], axis=1)
        transformation = TaperingZ()
        actual = transformation.transform(x, params)
        self.assertAlmostEqualNumpy(expected, actual)

    @parameterized.expand(zip(
        sample_points(),
        sample_params(num_params=2)
    ))
    def test_transformation_interval(self, x, params):
        transformation = TaperingZ()
        actual = transformation.transform(x, params)
        self.assertSound(actual, params, partial(transformation.transform, x))

    @parameterized.expand(zip(
        sample_points(),
        sample_params(num_params=2)
    ))
    def test_transformation_taylor(self, x, params):
        transformation = TaperingZ()
        actual = taylor.encode(transformation, x, params)
        self.assertSound(actual, params, partial(transformation.transform, x))


if __name__ == '__main__':
    unittest.main()
