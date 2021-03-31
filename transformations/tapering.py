from typing import List, Union

import numpy as np

import relaxations.interval as iv
from relaxations.interval import Interval
from transformations.transformation import Transformation


class TaperingZ(Transformation):

    def __init__(self):
        super().__init__(num_params=2)

    def transform(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[np.ndarray, Interval]:
        a, b = params
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        x_transformed = (0.5 * iv.square(a) * z + b * z + 1) * x
        y_transformed = (0.5 * iv.square(a) * z + b * z + 1) * y
        z_transformed = z
        return iv.stack([x_transformed, y_transformed, z_transformed], axis=1, convert=isinstance(a, Interval))

    def gradient_params(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[np.ndarray], List[Interval]]:
        a, b = params
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        return [
            iv.stack([(a * z) * x, (a * z) * y, iv.zeros_like(z)], axis=1, convert=isinstance(a, Interval)),
            iv.stack([z * x, z * y, iv.zeros_like(z)], axis=1, convert=isinstance(a, Interval))
        ]

    def gradient_points(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[np.ndarray], List[Interval]]:
        a, b = params
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        zero = iv.zeros_like(z)
        return [
            iv.stack([(0.5 * iv.square(a) * z + b * z + 1), zero, zero], axis=1, convert=isinstance(a, Interval)),
            iv.stack([zero, (0.5 * iv.square(a) * z + b * z + 1), zero], axis=1, convert=isinstance(a, Interval)),
            iv.stack([(0.5 * iv.square(a) + b) * x, (0.5 * iv.square(a) + b) * y, iv.ones_like(z)], axis=1, convert=isinstance(a, Interval)),
        ]

    def hessian_params(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[List[np.ndarray]], List[List[Interval]]]:
        a, b = params
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        zero = iv.zeros_like(x)
        return [
            [
                iv.stack([z * x, z * y, zero], axis=1, convert=isinstance(a, Interval)),
                iv.stack([zero, zero, zero], axis=1, convert=isinstance(a, Interval))
            ],
            [
                iv.stack([zero, zero, zero], axis=1, convert=isinstance(a, Interval)),
                iv.stack([zero, zero, zero], axis=1, convert=isinstance(a, Interval))
            ]
        ]

    def hessian_points(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[List[np.ndarray]], List[List[Interval]]]:
        a, b = params
        x = points[:, 0]
        zero = iv.zeros_like(x)
        one = iv.ones_like(x)
        return [
            [
                iv.stack([zero, zero, zero], axis=1, convert=isinstance(a, Interval)),
                iv.stack([zero, zero, zero], axis=1, convert=isinstance(a, Interval)),
                iv.stack([(0.5 * iv.square(a) + b) * one, zero, zero], axis=1, convert=isinstance(a, Interval)),
            ],
            [
                iv.stack([zero, zero, zero], axis=1, convert=isinstance(a, Interval)),
                iv.stack([zero, zero, zero], axis=1, convert=isinstance(a, Interval)),
                iv.stack([zero, (0.5 * iv.square(a) + b) * one, zero], axis=1, convert=isinstance(a, Interval)),
            ],
            [
                iv.stack([(0.5 * iv.square(a) + b) * one, zero, zero], axis=1, convert=isinstance(a, Interval)),
                iv.stack([zero, (0.5 * iv.square(a) + b) * one, zero], axis=1, convert=isinstance(a, Interval)),
                iv.stack([zero, zero, zero], axis=1, convert=isinstance(a, Interval)),
            ]
        ]

    def hessian_points_params(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[List[np.ndarray]], List[List[Interval]]]:
        a, b = params
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        zero = iv.zeros_like(x)
        return [
            [
                iv.stack([a * z, zero, zero], axis=1, convert=isinstance(a, Interval)),
                iv.stack([z, zero, zero], axis=1, convert=isinstance(a, Interval)),
            ],
            [
                iv.stack([zero, a * z, zero], axis=1, convert=isinstance(a, Interval)),
                iv.stack([zero, z, zero], axis=1, convert=isinstance(a, Interval)),
            ],
            [
                iv.stack([a * x, a * y, zero], axis=1, convert=isinstance(a, Interval)),
                iv.stack([x, y, zero], axis=1, convert=isinstance(a, Interval)),
            ],
        ]
