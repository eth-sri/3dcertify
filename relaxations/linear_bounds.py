from typing import Iterable

import numpy as np

from relaxations.interval import Interval

PI_HALF = np.pi / 2.0
PI = np.pi
TWO_PI = 2.0 * np.pi


class LinearBounds:

    def __init__(self, upper_slope: np.ndarray, upper_offset: np.ndarray,
                 lower_slope: np.ndarray, lower_offset: np.ndarray):
        self.upper_slope = upper_slope
        self.upper_offset = upper_offset
        self.lower_slope = lower_slope
        self.lower_offset = lower_offset

    def __neg__(self):
        return LinearBounds(-self.lower_slope, -self.lower_offset,
                            -self.upper_slope, -self.upper_offset)

    def __add__(self, other):
        if isinstance(other, LinearBounds):
            return LinearBounds(self.upper_slope + other.upper_slope,
                                self.upper_offset + other.upper_offset,
                                self.lower_slope + other.lower_slope,
                                self.lower_offset + other.lower_offset)
        elif isinstance(other, Interval):
            return LinearBounds(self.upper_slope, self.upper_offset + other.upper_bound,
                                self.lower_slope, self.lower_offset + other.lower_bound)
        else:
            return LinearBounds(self.upper_slope, self.upper_offset + other,
                                self.lower_slope, self.lower_offset + other)

    __radd__ = __add__

    def __mul__(self, other):
        if self.upper_slope.shape[-1] == 2:
            other_reshaped = np.expand_dims(other, axis=-1).repeat(2, axis=-1)
        elif self.upper_slope.shape[-1] == 3:
            other_reshaped = np.expand_dims(other, axis=-1).repeat(3, axis=-1)
        else:
            other_reshaped = other
        return LinearBounds(
            upper_slope=np.where(other_reshaped < 0.0, other_reshaped * self.lower_slope,
                                 other_reshaped * self.upper_slope),
            upper_offset=np.where(other < 0.0, other * self.lower_offset, other * self.upper_offset),
            lower_slope=np.where(other_reshaped < 0.0, other_reshaped * self.upper_slope,
                                 other_reshaped * self.lower_slope),
            lower_offset=np.where(other < 0.0, other * self.upper_offset, other * self.lower_offset)
        )

    def __rmul__(self, other):
        if other < 0.0:
            return LinearBounds(other * self.lower_slope, other * self.lower_offset,
                                other * self.upper_slope, other * self.upper_offset)
        else:
            return LinearBounds(other * self.upper_slope, other * self.upper_offset,
                                other * self.lower_slope, other * self.lower_offset)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def expand_shape(self, shape):
        upper_slope = np.empty(shape)
        upper_slope.fill(self.upper_slope.item())
        self.upper_slope = upper_slope

        upper_offset = np.empty(shape)
        upper_offset.fill(self.upper_offset.item())
        self.upper_offset = upper_offset

        lower_slope = np.empty(shape)
        lower_slope.fill(self.lower_slope.item())
        self.lower_slope = lower_slope

        lower_offset = np.empty(shape)
        lower_offset.fill(self.lower_offset.item())
        self.lower_offset = lower_offset

    def __repr__(self):
        return f"LinearBounds(upper_slope: {self.upper_slope}, upper_offset: {self.upper_offset}, " \
               f"lower_slope: {self.lower_slope}, lower_offset: {self.lower_offset})"

    def evaluate_at(self, x) -> Interval:
        return Interval(self.lower_offset + self.lower_slope[:, :, 0] * x, self.upper_offset + self.upper_slope[:, :, 0] * x)

    def evaluate_at_2d(self, x, y) -> Interval:
        return Interval(
            lower_bound=self.lower_offset + self.lower_slope[:, :, 0] * x + self.lower_slope[:, :, 1] * y,
            upper_bound=self.upper_offset + self.upper_slope[:, :, 0] * x + self.upper_slope[:, :, 1] * y,
        )


def stack_bounds(bounds: Iterable[LinearBounds], axis) -> LinearBounds:
    return LinearBounds(
        np.stack([b.upper_slope for b in bounds], axis),
        np.stack([b.upper_offset for b in bounds], axis),
        np.stack([b.lower_slope for b in bounds], axis),
        np.stack([b.lower_offset for b in bounds], axis),
    )
