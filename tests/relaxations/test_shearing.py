import unittest
from functools import partial

import numpy as np
from parameterized import parameterized

from relaxations import taylor
from tests.RelaxationTestCase import RelaxationTestCase, sample_points, sample_params
from transformations.shearing import ShearingZ


class TestShearingZ(RelaxationTestCase):

    @parameterized.expand(zip(
        sample_points(),
        np.random.uniform(-10, 10, (100, 2))
    ))
    def test_transformation_float(self, x, params):
        sx, sy = params
        transformed_x = x[:, 0] + sx * x[:, 2]
        transformed_y = x[:, 1] + sy * x[:, 2]
        transformed_z = x[:, 2]
        expected = np.stack([transformed_x, transformed_y, transformed_z], axis=1)
        transformation = ShearingZ()
        actual = transformation.transform(x, params)
        self.assertAlmostEqualNumpy(expected, actual)

    @parameterized.expand(zip(
        sample_points(),
        sample_params(num_params=2)
    ))
    def test_transformation_interval(self, x, params):
        transformation = ShearingZ()
        actual = transformation.transform(x, params)
        self.assertSound(actual, params, partial(transformation.transform, x))

    @parameterized.expand(zip(
        sample_points(),
        sample_params(num_params=2)
    ))
    def test_transformation_taylor(self, x, params):
        transformation = ShearingZ()
        actual = taylor.encode(transformation, x, params)
        self.assertSound(actual, params, partial(transformation.transform, x))


if __name__ == '__main__':
    unittest.main()
