from typing import Union, List

import numpy as np

from relaxations import interval as iv
from relaxations.interval import Interval
from transformations.transformation import Transformation


class RotationX(Transformation):

    def __init__(self):
        super().__init__(num_params=1)

    def transform(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[np.ndarray, Interval]:
        assert len(params) == 1
        alpha = params[0]
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        x_transformed = x
        y_transformed = iv.cos(alpha) * y - iv.sin(alpha) * z
        z_transformed = iv.sin(alpha) * y + iv.cos(alpha) * z
        return iv.stack([x_transformed, y_transformed, z_transformed], axis=1, convert=isinstance(alpha, Interval))

    def gradient_params(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[np.ndarray], List[Interval]]:
        (alpha,) = params
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        d_alpha_x = iv.zeros_like(x)
        d_alpha_y = -y * iv.sin(alpha) - z * iv.cos(alpha)
        d_alpha_z = y * iv.cos(alpha) - z * iv.sin(alpha)
        return [iv.stack([d_alpha_x, d_alpha_y, d_alpha_z], axis=1, convert=isinstance(alpha, Interval))]

    def gradient_points(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[np.ndarray], List[Interval]]:
        (alpha,) = params
        x = points[:, 0]
        zero = iv.zeros_like(x)
        one = iv.ones_like(x)
        d_x = iv.stack([
            one,
            zero,
            zero
        ], axis=1, convert=isinstance(alpha, Interval))
        d_y = iv.stack([
            zero,
            iv.cos(alpha) * one,
            iv.sin(alpha) * one
        ], axis=1, convert=isinstance(alpha, Interval))
        d_z = iv.stack([
            zero,
            -iv.sin(alpha) * one,
            iv.cos(alpha) * one
        ], axis=1, convert=isinstance(alpha, Interval))
        return [d_x, d_y, d_z]

    def hessian_params(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[List[np.ndarray]], List[List[Interval]]]:
        (alpha,) = params
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        d_alpha_alpha_x = iv.zeros_like(x)
        d_alpha_alpha_y = -y * iv.cos(alpha) + z * iv.sin(alpha)
        d_alpha_alpha_z = -y * iv.sin(alpha) - z * iv.cos(alpha)
        return [[iv.stack([d_alpha_alpha_x, d_alpha_alpha_y, d_alpha_alpha_z], axis=1, convert=isinstance(alpha, Interval))]]

    def hessian_points(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[List[np.ndarray]], List[List[Interval]]]:
        assert len(params) == 1
        zeros = iv.zeros_like(points)
        return [[zeros] * 3] * 3

    def hessian_points_params(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[List[np.ndarray]], List[List[Interval]]]:
        (alpha,) = params
        x = points[:, 0]
        zero = iv.zeros_like(x)
        one = iv.ones_like(x)
        d_xa = iv.zeros_like(points)
        d_ya = iv.stack([
            zero,
            -iv.sin(alpha) * one,
            iv.cos(alpha) * one
        ], axis=1, convert=isinstance(alpha, Interval))
        d_za = iv.stack([
            zero,
            -iv.cos(alpha) * one,
            -iv.sin(alpha) * one
        ], axis=1, convert=isinstance(alpha, Interval))
        return [[d_xa], [d_ya], [d_za]]


class RotationY(Transformation):

    def __init__(self):
        super().__init__(num_params=1)

    def transform(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[np.ndarray, Interval]:
        assert len(params) == 1
        alpha = params[0]
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        x_transformed = iv.cos(alpha) * x + iv.sin(alpha) * z
        y_transformed = y
        z_transformed = -iv.sin(alpha) * x + iv.cos(alpha) * z
        return iv.stack([x_transformed, y_transformed, z_transformed], axis=1, convert=isinstance(alpha, Interval))

    def gradient_params(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[np.ndarray], List[Interval]]:
        assert len(params) == 1
        alpha = params[0]
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        d_alpha_x = -x * iv.sin(alpha) + z * iv.cos(alpha)
        d_alpha_y = iv.zeros_like(y)
        d_alpha_z = -x * iv.cos(alpha) - z * iv.sin(alpha)
        return [iv.stack([d_alpha_x, d_alpha_y, d_alpha_z], axis=1, convert=isinstance(alpha, Interval))]

    def gradient_points(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[np.ndarray], List[Interval]]:
        (alpha,) = params
        x = points[:, 0]
        zero = iv.zeros_like(x)
        one = iv.ones_like(x)
        d_x = iv.stack([
            iv.cos(alpha) * one,
            zero,
            -iv.sin(alpha) * one
        ], axis=1, convert=isinstance(alpha, Interval))
        d_y = iv.stack([
            zero,
            one,
            zero
        ], axis=1, convert=isinstance(alpha, Interval))
        d_z = iv.stack([
            iv.sin(alpha) * one,
            zero,
            iv.cos(alpha) * one
        ], axis=1, convert=isinstance(alpha, Interval))
        return [d_x, d_y, d_z]

    def hessian_params(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[List[np.ndarray]], List[List[Interval]]]:
        assert len(params) == 1
        alpha = params[0]
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        d_alpha_alpha_x = -x * iv.cos(alpha) - z * iv.sin(alpha)
        d_alpha_alpha_y = iv.zeros_like(y)
        d_alpha_alpha_z = x * iv.sin(alpha) - z * iv.cos(alpha)
        return [[iv.stack([d_alpha_alpha_x, d_alpha_alpha_y, d_alpha_alpha_z], axis=1, convert=isinstance(alpha, Interval))]]

    def hessian_points(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[List[np.ndarray]], List[List[Interval]]]:
        assert len(params) == 1
        zeros = iv.zeros_like(points)
        return [[zeros] * 3] * 3

    def hessian_points_params(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[List[np.ndarray]], List[List[Interval]]]:
        (alpha,) = params
        x = points[:, 0]
        zero = iv.zeros_like(x)
        one = iv.ones_like(x)
        d_xa = iv.stack([
            -iv.sin(alpha) * one,
            zero,
            -iv.cos(alpha) * one
        ], axis=1, convert=isinstance(alpha, Interval))
        d_ya = iv.zeros_like(points)
        d_za = iv.stack([
            iv.cos(alpha) * one,
            zero,
            -iv.sin(alpha) * one
        ], axis=1, convert=isinstance(alpha, Interval))
        return [[d_xa], [d_ya], [d_za]]


class RotationZ(Transformation):

    def __init__(self):
        super().__init__(num_params=1)

    def transform(self, points: Union[np.ndarray, Interval], params: Union[List[float], List[Interval]]) -> Union[np.ndarray, Interval]:
        assert len(params) == 1
        alpha = params[0]
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        x_transformed = iv.cos(alpha) * x - iv.sin(alpha) * y
        y_transformed = iv.sin(alpha) * x + iv.cos(alpha) * y
        z_transformed = z
        return iv.stack([x_transformed, y_transformed, z_transformed], axis=1, convert=isinstance(alpha, Interval))

    def gradient_params(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[np.ndarray], List[Interval]]:
        assert len(params) == 1
        alpha = params[0]
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        d_alpha_x = -x * iv.sin(alpha) - y * iv.cos(alpha)
        d_alpha_y = x * iv.cos(alpha) - y * iv.sin(alpha)
        d_alpha_z = iv.zeros_like(z)
        return [iv.stack([d_alpha_x, d_alpha_y, d_alpha_z], axis=1, convert=isinstance(alpha, Interval))]

    def gradient_points(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[np.ndarray], List[Interval]]:
        (alpha,) = params
        x = points[:, 0]
        zero = iv.zeros_like(x)
        one = iv.ones_like(x)
        d_x = iv.stack([
            iv.cos(alpha) * one,
            iv.sin(alpha) * one,
            zero
        ], axis=1, convert=isinstance(alpha, Interval))
        d_y = iv.stack([
            -iv.sin(alpha) * one,
            iv.cos(alpha) * one,
            zero
        ], axis=1, convert=isinstance(alpha, Interval))
        d_z = iv.stack([
            zero,
            zero,
            one
        ], axis=1, convert=isinstance(alpha, Interval))
        return [d_x, d_y, d_z]

    def hessian_params(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[List[np.ndarray]], List[List[Interval]]]:
        assert len(params) == 1
        alpha = params[0]
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        d_alpha_alpha_x = -x * iv.cos(alpha) + y * iv.sin(alpha)
        d_alpha_alpha_y = -x * iv.sin(alpha) - y * iv.cos(alpha)
        d_alpha_alpha_z = iv.zeros_like(z)
        return [[iv.stack([d_alpha_alpha_x, d_alpha_alpha_y, d_alpha_alpha_z], axis=1, convert=isinstance(alpha, Interval))]]

    def hessian_points(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[List[np.ndarray]], List[List[Interval]]]:
        assert len(params) == 1
        zeros = iv.zeros_like(points)
        return [[zeros] * 3] * 3

    def hessian_points_params(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[List[np.ndarray]], List[List[Interval]]]:
        (alpha,) = params
        x = points[:, 0]
        zero = iv.zeros_like(x)
        one = iv.ones_like(x)
        d_xa = iv.stack([
            -iv.sin(alpha) * one,
            iv.cos(alpha) * one,
            zero
        ], axis=1, convert=isinstance(alpha, Interval))
        d_ya = iv.stack([
            -iv.cos(alpha) * one,
            -iv.sin(alpha) * one,
            zero
        ], axis=1, convert=isinstance(alpha, Interval))
        d_za = iv.zeros_like(points)
        return [[d_xa], [d_ya], [d_za]]
