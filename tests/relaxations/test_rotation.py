import unittest
from functools import partial

import numpy as np
from parameterized import parameterized
from scipy.spatial.transform import Rotation

from relaxations import taylor
from tests.RelaxationTestCase import RelaxationTestCase, sample_points, sample_params
from transformations.rotation import RotationZ, RotationX, RotationY
from util.rotation import rotate_z


class TestRotationX(RelaxationTestCase):

    @parameterized.expand(zip(
        sample_points(),
        np.random.uniform(-10, 10, (100, 1))
    ))
    def test_transformation_float(self, x, alpha):
        rotation = Rotation.from_euler('x', alpha.item())
        expected = x.dot(rotation.as_matrix().T)
        transformation = RotationX()
        actual = transformation.transform(x, alpha)
        self.assertAlmostEqualNumpy(expected, actual)

    @parameterized.expand(zip(
        sample_points(),
        sample_params(num_params=1)
    ))
    def test_transformation_interval(self, x, params):
        transformation = RotationX()
        actual = transformation.transform(x, params)
        self.assertSound(actual, params, partial(transformation.transform, x))

    @parameterized.expand(zip(
        sample_points(),
        sample_params(num_params=1)
    ))
    def test_transformation_taylor(self, x, params):
        transformation = RotationX()
        actual = taylor.encode(transformation, x, params)
        self.assertSound(actual, params, partial(transformation.transform, x))


class TestRotationY(RelaxationTestCase):

    @parameterized.expand(zip(
        sample_points(),
        np.random.uniform(-10, 10, (100, 1))
    ))
    def test_transformation_float(self, x, alpha):
        rotation = Rotation.from_euler('y', alpha.item())
        expected = x.dot(rotation.as_matrix().T)
        transformation = RotationY()
        actual = transformation.transform(x, alpha)
        self.assertAlmostEqualNumpy(expected, actual)

    @parameterized.expand(zip(
        sample_points(),
        sample_params(num_params=1)
    ))
    def test_transformation_interval(self, x, params):
        transformation = RotationY()
        actual = transformation.transform(x, params)
        self.assertSound(actual, params, partial(transformation.transform, x))

    @parameterized.expand(zip(
        sample_points(),
        sample_params(num_params=1)
    ))
    def test_transformation_taylor(self, x, params):
        transformation = RotationY()
        actual = taylor.encode(transformation, x, params)
        self.assertSound(actual, params, partial(transformation.transform, x))


class TestRotationZ(RelaxationTestCase):

    @parameterized.expand(zip(
        sample_points(),
        np.random.uniform(-10, 10, (100, 1))
    ))
    def test_transformation_float(self, x, alpha):
        transformation = RotationZ()
        expected = rotate_z(x, alpha.item())
        actual = transformation.transform(x, alpha)
        self.assertAlmostEqualNumpy(expected, actual)

    @parameterized.expand(zip(
        sample_points(),
        sample_params(num_params=1)
    ))
    def test_transformation_interval(self, x, params):
        transformation = RotationZ()
        actual = transformation.transform(x, params)
        self.assertSound(actual, params, partial(transformation.transform, x))

    @parameterized.expand(zip(
        sample_points(),
        sample_params(num_params=1)
    ))
    def test_transformation_taylor(self, x, params):
        transformation = RotationZ()
        actual = taylor.encode(transformation, x, params)
        self.assertSound(actual, params, partial(transformation.transform, x))


if __name__ == '__main__':
    unittest.main()
