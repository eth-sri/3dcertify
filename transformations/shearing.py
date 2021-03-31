from typing import Union, List

import numpy as np

import relaxations.interval as iv
from relaxations.interval import Interval
from transformations.transformation import Transformation


class ShearingZ(Transformation):

    def __init__(self):
        super().__init__(num_params=2)

    def transform(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[np.ndarray, Interval]:
        sx, sy = params
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        x_transformed = sx * z + x
        y_transformed = sy * z + y
        z_transformed = z
        return iv.stack([x_transformed, y_transformed, z_transformed], axis=1, convert=isinstance(sx, Interval))

    def gradient_params(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[np.ndarray], List[Interval]]:
        sx, sy = params
        z = points[:, 2]
        zero = iv.zeros_like(z)
        return [
            iv.stack([z, zero, zero], axis=1, convert=isinstance(sx, Interval)),
            iv.stack([zero, z, zero], axis=1, convert=isinstance(sx, Interval))
        ]

    def gradient_points(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[np.ndarray], List[Interval]]:
        sx, sy = params
        z = points[:, 2]
        zero = iv.zeros_like(z)
        one = iv.ones_like(z)
        return [
            iv.stack([one, zero, zero], axis=1, convert=isinstance(sx, Interval)),
            iv.stack([zero, one, zero], axis=1, convert=isinstance(sx, Interval)),
            iv.stack([sx * one, sy * one, one], axis=1, convert=isinstance(sx, Interval)),
        ]

    def hessian_params(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[List[np.ndarray]], List[List[Interval]]]:
        assert len(params) == 2
        zero = iv.as_interval(iv.zeros_like(points)) if isinstance(params[0], Interval) else iv.zeros_like(points)
        return [[zero, zero], [zero, zero]]

    def hessian_points(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[List[np.ndarray]], List[List[Interval]]]:
        assert len(params) == 2
        zero = iv.as_interval(iv.zeros_like(points)) if isinstance(params[0], Interval) else iv.zeros_like(points)
        return [[zero, zero, zero], [zero, zero, zero], [zero, zero, zero]]

    def hessian_points_params(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[List[np.ndarray]], List[List[Interval]]]:
        sx, sy = params
        x = points[:, 0]
        zero = iv.zeros_like(x)
        one = iv.ones_like(x)
        return [
            [
                iv.stack([zero, zero, zero], axis=1, convert=isinstance(sx, Interval)),
                iv.stack([zero, zero, zero], axis=1, convert=isinstance(sx, Interval)),
            ],
            [
                iv.stack([zero, zero, zero], axis=1, convert=isinstance(sx, Interval)),
                iv.stack([zero, zero, zero], axis=1, convert=isinstance(sx, Interval)),
            ],
            [
                iv.stack([one, zero, zero], axis=1, convert=isinstance(sx, Interval)),
                iv.stack([zero, one, zero], axis=1, convert=isinstance(sx, Interval)),
            ],
        ]
