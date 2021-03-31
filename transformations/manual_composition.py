from typing import Union, List

import numpy as np

from relaxations import interval as iv
from relaxations.interval import Interval
from transformations.rotation import RotationZ
from transformations.tapering import TaperingZ
from transformations.transformation import Transformation


class TaperingRotation(Transformation):

    def __init__(self):
        super().__init__(3)

    def transform(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[np.ndarray, Interval]:
        a, b, theta = params
        return TaperingZ().transform(RotationZ().transform(points, [theta]), [a, b])

    def gradient_params(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[np.ndarray], List[Interval]]:
        a, b, theta = params
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        zero = iv.zeros_like(z)
        return [
            iv.stack([
                (a * z) * (iv.cos(theta) * x - iv.sin(theta) * y),
                (a * z) * (iv.sin(theta) * x + iv.cos(theta) * y),
                zero
            ], axis=1, convert=isinstance(a, Interval)),
            iv.stack([
                z * (iv.cos(theta) * x - iv.sin(theta) * y),
                z * (iv.sin(theta) * x + iv.cos(theta) * y),
                zero
            ], axis=1, convert=isinstance(a, Interval)),
            iv.stack([
                (0.5 * iv.square(a) * z + b * z + 1) * (-iv.sin(theta) * x - iv.cos(theta) * y),
                (0.5 * iv.square(a) * z + b * z + 1) * (iv.cos(theta) * x - iv.sin(theta) * y),
                zero
            ], axis=1, convert=isinstance(a, Interval)),
        ]

    def gradient_points(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[np.ndarray], List[Interval]]:
        a, b, theta = params
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        zero = iv.zeros_like(z)
        one = iv.ones_like(z)
        return [
            iv.stack([
                (0.5 * iv.square(a) * z + b * z + 1) * iv.cos(theta),
                (0.5 * iv.square(a) * z + b * z + 1) * iv.sin(theta),
                zero
            ], axis=1, convert=isinstance(a, Interval)),
            iv.stack([
                -(0.5 * iv.square(a) * z + b * z + 1) * iv.sin(theta),
                (0.5 * iv.square(a) * z + b * z + 1) * iv.cos(theta),
                zero
            ], axis=1, convert=isinstance(a, Interval)),
            iv.stack([
                (0.5 * iv.square(a) + b) * (iv.cos(theta) * x - iv.sin(theta) * y),
                (0.5 * iv.square(a) + b) * (iv.sin(theta) * x + iv.cos(theta) * y),
                one
            ], axis=1, convert=isinstance(a, Interval))
        ]

    def hessian_params(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[List[np.ndarray]], List[List[Interval]]]:
        a, b, theta = params
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        zero = iv.zeros_like(z)
        return [
            [
                iv.stack([
                    z * (iv.cos(theta) * x - iv.sin(theta) * y),
                    z * (iv.sin(theta) * x + iv.cos(theta) * y),
                    zero
                ], axis=1, convert=isinstance(a, Interval)),
                iv.stack([
                    zero,
                    zero,
                    zero
                ], axis=1, convert=isinstance(a, Interval)),
                iv.stack([
                    -a * z * (iv.sin(theta) * x + iv.cos(theta) * y),
                    a * z * (iv.cos(theta) * x - iv.sin(theta) * y),
                    zero
                ], axis=1, convert=isinstance(a, Interval))
            ],
            [
                iv.stack([
                    zero,
                    zero,
                    zero
                ], axis=1, convert=isinstance(a, Interval)),
                iv.stack([
                    zero,
                    zero,
                    zero
                ], axis=1, convert=isinstance(a, Interval)),
                iv.stack([
                    -z * (iv.sin(theta) * x + iv.cos(theta) * y),
                    z * (iv.cos(theta) * x - iv.sin(theta) * y),
                    zero
                ], axis=1, convert=isinstance(a, Interval))
            ],
            [
                iv.stack([
                    -a * z * (iv.sin(theta) * x + iv.cos(theta) * y),
                    a * z * (iv.cos(theta) * x - iv.sin(theta) * y),
                    zero
                ], axis=1, convert=isinstance(a, Interval)),
                iv.stack([
                    -z * (iv.sin(theta) * x + iv.cos(theta) * y),
                    z * (iv.cos(theta) * x - iv.sin(theta) * y),
                    zero
                ], axis=1, convert=isinstance(a, Interval)),
                iv.stack([
                    (0.5 * iv.square(a) * z + b * z + 1) * (-iv.cos(theta) * x + iv.sin(theta) * y),
                    (0.5 * iv.square(a) * z + b * z + 1) * (-iv.sin(theta) * x - iv.cos(theta) * y),
                    zero
                ], axis=1, convert=isinstance(a, Interval))
            ]
        ]

    def hessian_points(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[List[np.ndarray]], List[List[Interval]]]:
        a, b, theta = params
        z = points[:, 2]
        zero = iv.zeros_like(z)
        one = iv.ones_like(z)
        return [
            [
                iv.stack([
                    zero,
                    zero,
                    zero
                ], axis=1, convert=isinstance(a, Interval)),
                iv.stack([
                    zero,
                    zero,
                    zero
                ], axis=1, convert=isinstance(a, Interval)),
                iv.stack([
                    (0.5 * iv.square(a) + b) * iv.cos(theta) * one,
                    (0.5 * iv.square(a) + b) * iv.sin(theta) * one,
                    zero
                ], axis=1, convert=isinstance(a, Interval)),
            ],
            [
                iv.stack([
                    zero,
                    zero,
                    zero
                ], axis=1, convert=isinstance(a, Interval)),
                iv.stack([
                    zero,
                    zero,
                    zero
                ], axis=1, convert=isinstance(a, Interval)),
                iv.stack([
                    -(0.5 * iv.square(a) + b) * iv.sin(theta) * one,
                    (0.5 * iv.square(a) + b) * iv.cos(theta) * one,
                    zero
                ], axis=1, convert=isinstance(a, Interval))
            ],
            [
                iv.stack([
                    (0.5 * iv.square(a) + b) * iv.cos(theta) * one,
                    (0.5 * iv.square(a) + b) * iv.sin(theta) * one,
                    zero
                ], axis=1, convert=isinstance(a, Interval)),
                iv.stack([
                    -(0.5 * iv.square(a) + b) * iv.sin(theta) * one,
                    (0.5 * iv.square(a) + b) * iv.cos(theta) * one,
                    zero
                ], axis=1, convert=isinstance(a, Interval)),
                iv.stack([
                    zero,
                    zero,
                    zero
                ], axis=1, convert=isinstance(a, Interval))
            ],
        ]

    def hessian_points_params(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[List[np.ndarray]], List[List[Interval]]]:
        a, b, theta = params
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        zero = iv.zeros_like(z)
        return [
            [
                iv.stack([
                    a * z * iv.cos(theta),
                    a * z * iv.sin(theta),
                    zero
                ], axis=1, convert=isinstance(a, Interval)),
                iv.stack([
                    z * iv.cos(theta),
                    z * iv.sin(theta),
                    zero
                ], axis=1, convert=isinstance(a, Interval)),
                iv.stack([
                    -(0.5 * iv.square(a) * z + b * z + 1) * iv.sin(theta),
                    (0.5 * iv.square(a) * z + b * z + 1) * iv.cos(theta),
                    zero
                ], axis=1, convert=isinstance(a, Interval)),
            ],
            [
                iv.stack([
                    -a * z * iv.sin(theta),
                    a * z * iv.cos(theta),
                    zero
                ], axis=1, convert=isinstance(a, Interval)),
                iv.stack([
                    -z * iv.sin(theta),
                    z * iv.cos(theta),
                    zero
                ], axis=1, convert=isinstance(a, Interval)),
                iv.stack([
                    -(0.5 * iv.square(a) * z + b * z + 1) * iv.cos(theta),
                    -(0.5 * iv.square(a) * z + b * z + 1) * iv.sin(theta),
                    zero
                ], axis=1, convert=isinstance(a, Interval))
            ],
            [
                iv.stack([
                    a * (iv.cos(theta) * x - iv.sin(theta) * y),
                    a * (iv.sin(theta) * x + iv.cos(theta) * y),
                    zero
                ], axis=1, convert=isinstance(a, Interval)),
                iv.stack([
                    iv.cos(theta) * x - iv.sin(theta) * y,
                    iv.sin(theta) * x + iv.cos(theta) * y,
                    zero
                ], axis=1, convert=isinstance(a, Interval)),
                iv.stack([
                    (0.5 * iv.square(a) + b) * (-iv.sin(theta) * x - iv.cos(theta) * y),
                    (0.5 * iv.square(a) + b) * (iv.cos(theta) * x - iv.sin(theta) * y),
                    zero
                ], axis=1, convert=isinstance(a, Interval))
            ],
        ]


class RotationZX(Transformation):

    def __init__(self):
        super().__init__(2)

    def transform(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[np.ndarray, Interval]:
        a, b = params
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        return iv.stack([
            iv.cos(a) * x - iv.sin(a) * iv.cos(b) * y + iv.sin(a) * iv.sin(b) * z,
            iv.sin(a) * x + iv.cos(a) * iv.cos(b) * y - iv.cos(a) * iv.sin(b) * z,
            iv.sin(b) * y + iv.cos(b) * z
        ], axis=1, convert=isinstance(a, Interval))

    def gradient_params(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[np.ndarray], List[Interval]]:
        a, b = params
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        return [
            iv.stack([
                -iv.sin(a) * x - iv.cos(a) * iv.cos(b) * y + iv.cos(a) * iv.sin(b) * z,
                iv.cos(a) * x - iv.sin(a) * iv.cos(b) * y + iv.sin(a) * iv.sin(b) * z,
                iv.zeros_like(z)
            ], axis=1, convert=isinstance(a, Interval)),
            iv.stack([
                iv.sin(a) * iv.sin(b) * y + iv.sin(a) * iv.cos(b) * z,
                -iv.cos(a) * iv.sin(b) * y - iv.cos(a) * iv.cos(b) * z,
                iv.cos(b) * y - iv.sin(b) * z
            ], axis=1, convert=isinstance(a, Interval)),
        ]

    def gradient_points(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[np.ndarray], List[Interval]]:
        a, b = params
        x = points[:, 0]
        zero = iv.zeros_like(x)
        one = iv.ones_like(x)
        return [
            iv.stack([
                iv.cos(a) * one,
                iv.sin(a) * one,
                zero
            ], axis=1, convert=isinstance(a, Interval)),
            iv.stack([
                -iv.sin(a) * iv.cos(b) * one,
                iv.cos(a) * iv.cos(b) * one,
                iv.sin(b) * one
            ], axis=1, convert=isinstance(a, Interval)),
            iv.stack([
                iv.sin(a) * iv.sin(b) * one,
                -iv.cos(a) * iv.sin(b) * one,
                iv.cos(b) * one
            ], axis=1, convert=isinstance(a, Interval)),
        ]

    def hessian_params(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[List[np.ndarray]], List[List[Interval]]]:
        a, b = params
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        zero = iv.zeros_like(x)
        return [
            [
                iv.stack([
                    -iv.cos(a) * x + iv.sin(a) * iv.cos(b) * y - iv.sin(a) * iv.sin(b) * z,
                    -iv.sin(a) * x - iv.cos(a) * iv.cos(b) * y + iv.cos(a) * iv.sin(b) * z,
                    zero
                ], axis=1, convert=isinstance(a, Interval)),
                iv.stack([
                    iv.cos(a) * iv.sin(b) * y + iv.cos(a) * iv.cos(b) * z,
                    iv.sin(a) * iv.sin(b) * y + iv.sin(a) * iv.cos(b) * z,
                    zero
                ], axis=1, convert=isinstance(a, Interval)),
            ],
            [
                iv.stack([
                    iv.cos(a) * iv.sin(b) * y + iv.cos(a) * iv.cos(b) * z,
                    iv.sin(a) * iv.sin(b) * y + iv.sin(a) * iv.cos(b) * z,
                    zero
                ], axis=1, convert=isinstance(a, Interval)),
                iv.stack([
                    iv.sin(a) * iv.cos(b) * y - iv.sin(a) * iv.sin(b) * z,
                    -iv.cos(a) * iv.cos(b) * y + iv.cos(a) * iv.sin(b) * z,
                    -iv.sin(b) * y - iv.cos(b) * z
                ], axis=1, convert=isinstance(a, Interval)),
            ]
        ]

    def hessian_points(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[List[np.ndarray]], List[List[Interval]]]:
        a, b = params
        zero = iv.as_interval(iv.zeros_like(points)) if isinstance(a, Interval) else iv.zeros_like(points)
        return [[zero] * 3 for _ in range(3)]

    def hessian_points_params(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[List[np.ndarray]], List[List[Interval]]]:
        a, b = params
        x = points[:, 0]
        zero = iv.zeros_like(x)
        one = iv.ones_like(x)
        return [
            [
                iv.stack([
                    -iv.sin(a) * one,
                    iv.cos(a) * one,
                    zero
                ], axis=1, convert=isinstance(a, Interval)),
                iv.stack([
                    zero,
                    zero,
                    zero
                ], axis=1, convert=isinstance(a, Interval)),
            ],
            [
                iv.stack([
                    -iv.cos(a) * iv.cos(b) * one,
                    -iv.sin(a) * iv.cos(b) * one,
                    zero
                ], axis=1, convert=isinstance(a, Interval)),
                iv.stack([
                    iv.sin(a) * iv.sin(b) * one,
                    -iv.cos(a) * iv.sin(b) * one,
                    iv.cos(b) * one
                ], axis=1, convert=isinstance(a, Interval)),
            ],
            [
                iv.stack([
                    iv.cos(a) * iv.sin(b) * one,
                    iv.sin(a) * iv.sin(b) * one,
                    zero
                ], axis=1, convert=isinstance(a, Interval)),
                iv.stack([
                    iv.sin(a) * iv.cos(b) * one,
                    -iv.cos(a) * iv.cos(b) * one,
                    -iv.sin(b) * one
                ], axis=1, convert=isinstance(a, Interval)),
            ]
        ]
