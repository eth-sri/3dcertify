import itertools
from typing import List

import numpy as np

import relaxations.interval as iv
from relaxations.interval import Interval
from relaxations.linear_bounds import LinearBounds
from transformations.transformation import Transformation


def encode(transformation: Transformation, points: np.ndarray, params: List[Interval]) -> LinearBounds:
    center = [p.center() for p in params]
    zero_order_term = transformation.transform(points, center)
    gradients = transformation.gradient_params(points, center)
    hessian = transformation.hessian_params(points, params)

    offset = zero_order_term
    for i in range(len(params)):
        offset = offset - (gradients[i] * center[i])

    second_order_bounds = 0.0
    for i, j in itertools.product(range(len(params)), range(len(params))):
        factor = iv.square(params[i] - center[i]) if i == j else (params[i] - center[i]) * (params[j] - center[j])
        second_order_bounds = second_order_bounds + (0.5 * hessian[i][j] * factor)

    return LinearBounds(
        upper_slope=np.stack(gradients, axis=2),
        lower_slope=np.stack(gradients, axis=2),
        upper_offset=offset + second_order_bounds.upper_bound,
        lower_offset=offset + second_order_bounds.lower_bound,
    )
