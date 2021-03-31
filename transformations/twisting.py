from typing import Union, List

import numpy as np

import relaxations.interval as iv
from relaxations.interval import Interval
from transformations.transformation import Transformation


class TwistingZ(Transformation):

    def __init__(self):
        super().__init__(num_params=1)

    def transform(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[np.ndarray, Interval]:
        assert len(params) == 1
        alpha = params[0]
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        x_transformed = x * iv.cos(alpha * z) - y * iv.sin(alpha * z)
        y_transformed = x * iv.sin(alpha * z) + y * iv.cos(alpha * z)
        z_transformed = z
        return iv.stack([x_transformed, y_transformed, z_transformed], axis=1, convert=isinstance(alpha, Interval))

    def gradient_params(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[np.ndarray], List[Interval]]:
        assert len(params) == 1
        alpha = params[0]
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        d_alpha_x = -z * (x * iv.sin(z * alpha) + y * iv.cos(z * alpha))
        d_alpha_y = z * (x * iv.cos(z * alpha) - y * iv.sin(z * alpha))
        d_alpha_z = iv.zeros_like(z)
        return [iv.stack([d_alpha_x, d_alpha_y, d_alpha_z], axis=1, convert=isinstance(alpha, Interval))]

    def gradient_points(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[np.ndarray], List[Interval]]:
        (alpha,) = params
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        zero = iv.zeros_like(z)
        one = iv.ones_like(z)
        return [
            iv.stack([
                iv.cos(alpha * z),
                iv.sin(alpha * z),
                zero
            ], axis=1, convert=isinstance(alpha, Interval)),
            iv.stack([
                -iv.sin(alpha * z),
                iv.cos(alpha * z),
                zero
            ], axis=1, convert=isinstance(alpha, Interval)),
            iv.stack([
                alpha * (-iv.sin(alpha * z) * x - iv.cos(alpha * z) * y),
                alpha * (iv.cos(alpha * z) * x - iv.sin(alpha * z) * y),
                one
            ], axis=1, convert=isinstance(alpha, Interval)),
        ]

    def hessian_params(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[List[np.ndarray]], List[List[Interval]]]:
        assert len(params) == 1
        alpha = params[0]
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        d_alpha_alpha_x = -iv.square(z) * (x * iv.cos(alpha * z) - y * iv.sin(alpha * z))
        d_alpha_alpha_y = -iv.square(z) * (x * iv.sin(alpha * z) + y * iv.cos(alpha * z))
        d_alpha_alpha_z = iv.zeros_like(z)
        return [[iv.stack([d_alpha_alpha_x, d_alpha_alpha_y, d_alpha_alpha_z], axis=1, convert=isinstance(alpha, Interval))]]

    def hessian_points(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[List[np.ndarray]], List[List[Interval]]]:
        (alpha,) = params
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        zero = iv.zeros_like(x)
        return [
            [
                iv.stack([
                    zero,
                    zero,
                    zero,
                ], axis=1, convert=isinstance(alpha, Interval)),
                iv.stack([
                    zero,
                    zero,
                    zero,
                ], axis=1, convert=isinstance(alpha, Interval)),
                iv.stack([
                    -alpha * iv.sin(alpha * z),
                    alpha * iv.cos(alpha * z),
                    zero,
                ], axis=1, convert=isinstance(alpha, Interval)),
            ],
            [
                iv.stack([
                    zero,
                    zero,
                    zero,
                ], axis=1, convert=isinstance(alpha, Interval)),
                iv.stack([
                    zero,
                    zero,
                    zero,
                ], axis=1, convert=isinstance(alpha, Interval)),
                iv.stack([
                    -alpha * iv.cos(alpha * z),
                    -alpha * iv.sin(alpha * z),
                    zero,
                ], axis=1, convert=isinstance(alpha, Interval)),
            ],
            [
                iv.stack([
                    -alpha * iv.sin(alpha * z),
                    alpha * iv.cos(alpha * z),
                    zero,
                ], axis=1, convert=isinstance(alpha, Interval)),
                iv.stack([
                    -alpha * iv.cos(alpha * z),
                    -alpha * iv.sin(alpha * z),
                    zero,
                ], axis=1, convert=isinstance(alpha, Interval)),
                iv.stack([
                    iv.square(alpha) * (-iv.cos(alpha * z) * x + iv.sin(alpha * z) * y),
                    iv.square(alpha) * (-iv.sin(alpha * z) * x - iv.cos(alpha * z) * y),
                    zero,
                ], axis=1, convert=isinstance(alpha, Interval)),
            ],
        ]

    def hessian_points_params(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[List[np.ndarray]], List[List[Interval]]]:
        (alpha,) = params
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        zero = iv.zeros_like(x)
        return [
            [
                iv.stack([
                    -z * iv.sin(alpha * z),
                    z * iv.cos(alpha * z),
                    zero,
                ], axis=1, convert=isinstance(alpha, Interval))
            ],
            [
                iv.stack([
                    -z * iv.cos(alpha * z),
                    -z * iv.sin(alpha * z),
                    zero,
                ], axis=1, convert=isinstance(alpha, Interval))
            ],
            [
                iv.stack([
                    -x * iv.sin(alpha * z) - alpha * x * z * iv.cos(alpha * z) + alpha * y * z * iv.sin(alpha * z) - y * iv.cos(alpha * z),
                    -alpha * x * z * iv.sin(alpha * z) + x * iv.cos(alpha * z) - y * iv.sin(alpha * z) - alpha * y * z * iv.cos(alpha * z),
                    zero,
                ], axis=1, convert=isinstance(alpha, Interval))
            ],
        ]
