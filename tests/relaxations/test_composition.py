import unittest
from functools import partial

import numpy as np
from parameterized import parameterized

from relaxations import taylor
from tests.RelaxationTestCase import RelaxationTestCase, sample_points, sample_params
from transformations.composition import Composition
from transformations.manual_composition import RotationZX, TaperingRotation
from transformations.rotation import RotationZ, RotationX, RotationY
from transformations.tapering import TaperingZ
from transformations.twisting import TwistingZ


class TestCompositionRotZX(RelaxationTestCase):

    @parameterized.expand(zip(
        sample_points(),
        np.random.uniform(-10, 10, (100, 2))
    ))
    def test_transformation_float(self, points, params):
        manual_transformation = RotationZX()
        auto_transformation = Composition(RotationZ(), RotationX())
        expected = manual_transformation.transform(points, params)
        actual = auto_transformation.transform(points, params)
        self.assertAlmostEqualNumpy(expected, actual)

    @parameterized.expand(zip(
        sample_points(),
        sample_params(2)
    ))
    def test_transformation_interval(self, points, params):
        manual_transformation = RotationZX()
        auto_transformation = Composition(RotationZ(), RotationX())
        self.assertSound(manual_transformation.transform(points, params), params, partial(manual_transformation.transform, points))
        self.assertSound(auto_transformation.transform(points, params), params, partial(auto_transformation.transform, points))

    @parameterized.expand(zip(
        sample_points(),
        np.random.uniform(-10, 10, (100, 2))
    ))
    def test_gradient_params_float(self, points, params):
        expected = RotationZX().gradient_params(points, params)
        actual = Composition(RotationZ(), RotationX()).gradient_params(points, params)
        self.assertAlmostEqualList(expected, actual)

    @parameterized.expand(zip(
        sample_points(),
        np.random.uniform(-10, 10, (100, 2))
    ))
    def test_gradient_points_float(self, points, params):
        expected = RotationZX().gradient_points(points, params)
        actual = Composition(RotationZ(), RotationX()).gradient_points(points, params)
        self.assertAlmostEqualList(expected, actual)

    @parameterized.expand(zip(
        sample_points(),
        np.random.uniform(-10, 10, (100, 2))
    ))
    def test_hessian_params_float(self, points, params):
        expected = RotationZX().hessian_params(points, params)
        actual = Composition(RotationZ(), RotationX()).hessian_params(points, params)
        self.assertAlmostEqualList(expected, actual)

    @parameterized.expand(zip(
        sample_points(),
        np.random.uniform(-10, 10, (100, 2))
    ))
    def test_hessian_points_float(self, points, params):
        expected = RotationZX().hessian_points(points, params)
        actual = Composition(RotationZ(), RotationX()).hessian_points(points, params)
        self.assertAlmostEqualList(expected, actual)

    @parameterized.expand(zip(
        sample_points(),
        np.random.uniform(-10, 10, (100, 2))
    ))
    def test_hessian_points_params_float(self, points, params):
        expected = RotationZX().hessian_points_params(points, params)
        actual = Composition(RotationZ(), RotationX()).hessian_points_params(points, params)
        self.assertAlmostEqualList(expected, actual)

    @parameterized.expand(zip(
        sample_points(),
        sample_params(num_params=2)
    ))
    def test_taylor_approximation(self, x, params):
        manual_transformation = RotationZX()
        auto_transformation = Composition(RotationZ(), RotationX())
        manual = taylor.encode(manual_transformation, x, params)
        auto = taylor.encode(auto_transformation, x, params)
        self.assertSound(manual, params, partial(manual_transformation.transform, x))
        self.assertSound(auto, params, partial(auto_transformation.transform, x))


class TestCompositionSoundness(RelaxationTestCase):

    @parameterized.expand(zip(
        sample_points(),
        sample_params(num_params=3)
    ))
    def test_taylor_twisting_tapering(self, x, params):
        transformation = Composition(TwistingZ(), TaperingZ())
        actual = taylor.encode(transformation, x, params)
        self.assertSound(actual, params, partial(transformation.transform, x))

    @parameterized.expand(zip(
        sample_points(),
        sample_params(num_params=2)
    ))
    def test_taylor_twisting_rotation(self, x, params):
        transformation = Composition(TwistingZ(), RotationZ())
        actual = taylor.encode(transformation, x, params)
        self.assertSound(actual, params, partial(transformation.transform, x))

    @parameterized.expand(zip(
        sample_points(),
        sample_params(num_params=4)
    ))
    def test_taylor_twisting_rotation_tapering(self, x, params):
        transformation = Composition(TwistingZ(), Composition(RotationZ(), TaperingZ()))
        actual = taylor.encode(transformation, x, params)
        self.assertSound(actual, params, partial(transformation.transform, x))

    @parameterized.expand(zip(
        sample_points(),
        sample_params(num_params=3)
    ))
    def test_taylor_rot_zyx(self, x, params):
        transformation = Composition(RotationZ(), Composition(RotationY(), RotationX()))
        actual = taylor.encode(transformation, x, params)
        self.assertSound(actual, params, partial(transformation.transform, x))


class TestTaperingRotation(RelaxationTestCase):

    @parameterized.expand(zip(
        sample_points(),
        np.random.uniform(-10, 10, (100, 3))
    ))
    def test_transformation_float(self, points, params):
        expected = TaperingRotation().transform(points, params)
        actual = Composition(TaperingZ(), RotationZ()).transform(points, params)
        self.assertAlmostEqualNumpy(expected, actual)

    @parameterized.expand(zip(
        sample_points(),
        sample_params(3)
    ))
    def test_transformation_interval(self, points, params):
        expected = TaperingRotation().transform(points, params)
        actual = Composition(TaperingZ(), RotationZ()).transform(points, params)
        self.assertAlmostEqualInterval(expected, actual)

    @parameterized.expand(zip(
        sample_points(),
        np.random.uniform(-10, 10, (100, 3))
    ))
    def test_gradient_params_float(self, points, params):
        expected = TaperingRotation().gradient_params(points, params)
        actual = Composition(TaperingZ(), RotationZ()).gradient_params(points, params)
        self.assertAlmostEqualList(expected, actual)

    @parameterized.expand(zip(
        sample_points(),
        sample_params(3)
    ))
    def test_gradient_params_interval(self, points, params):
        expected = TaperingRotation().gradient_params(points, params)
        actual = Composition(TaperingZ(), RotationZ()).gradient_params(points, params)
        self.assertAlmostEqualList(expected, actual)

    @parameterized.expand(zip(
        sample_points(),
        np.random.uniform(-10, 10, (100, 3))
    ))
    def test_gradient_points_float(self, points, params):
        expected = TaperingRotation().gradient_points(points, params)
        actual = Composition(TaperingZ(), RotationZ()).gradient_points(points, params)
        self.assertAlmostEqualList(expected, actual)

    @parameterized.expand(zip(
        sample_points(),
        sample_params(3)
    ))
    def test_gradient_points_interval(self, points, params):
        expected = TaperingRotation().gradient_points(points, params)
        actual = Composition(TaperingZ(), RotationZ()).gradient_points(points, params)
        self.assertAlmostEqualList(expected, actual)

    @parameterized.expand(zip(
        sample_points(),
        np.random.uniform(-10, 10, (100, 3))
    ))
    def test_hessian_params_float(self, points, params):
        expected = TaperingRotation().hessian_params(points, params)
        actual = Composition(TaperingZ(), RotationZ()).hessian_params(points, params)
        self.assertAlmostEqualList(expected, actual)

    @parameterized.expand(zip(
        sample_points(),
        sample_params(3)
    ))
    def test_hessian_params_interval(self, points, params):
        expected = TaperingRotation().hessian_params(points, params)
        actual = Composition(TaperingZ(), RotationZ()).hessian_params(points, params)
        self.assertAlmostEqualList(expected, actual)

    @parameterized.expand(zip(
        sample_points(),
        np.random.uniform(-10, 10, (100, 3))
    ))
    def test_hessian_points_float(self, points, params):
        expected = TaperingRotation().hessian_points(points, params)
        actual = Composition(TaperingZ(), RotationZ()).hessian_points(points, params)
        self.assertAlmostEqualList(expected, actual)

    @parameterized.expand(zip(
        sample_points(),
        sample_params(3)
    ))
    def test_hessian_points_interval(self, points, params):
        expected = TaperingRotation().hessian_points(points, params)
        actual = Composition(TaperingZ(), RotationZ()).hessian_points(points, params)
        self.assertAlmostEqualList(expected, actual)

    @parameterized.expand(zip(
        sample_points(),
        np.random.uniform(-10, 10, (100, 3))
    ))
    def test_hessian_points_params_float(self, points, params):
        expected = TaperingRotation().hessian_points_params(points, params)
        actual = Composition(TaperingZ(), RotationZ()).hessian_points_params(points, params)
        self.assertAlmostEqualList(expected, actual)

    @parameterized.expand(zip(
        sample_points(),
        sample_params(3)
    ))
    def test_hessian_points_params_interval(self, points, params):
        expected = TaperingRotation().hessian_points_params(points, params)
        actual = Composition(TaperingZ(), RotationZ()).hessian_points_params(points, params)
        self.assertAlmostEqualList(expected, actual)

    @parameterized.expand(zip(
        sample_points(),
        sample_params(num_params=3)
    ))
    def test_taylor_approximation(self, x, params):
        manual_transformation = TaperingRotation()
        auto_transformation = Composition(TaperingZ(), RotationZ())
        expected = taylor.encode(manual_transformation, x, params)
        actual = taylor.encode(auto_transformation, x, params)

        self.assertSound(expected, params, partial(manual_transformation.transform, x))
        self.assertSound(actual, params, partial(auto_transformation.transform, x))
        self.assertAlmostEqualBounds(expected, actual)


if __name__ == '__main__':
    unittest.main()
