from typing import Iterable, Union

import numpy as np

PI_HALF = np.pi / 2.0
PI = np.pi
TWO_PI = 2.0 * np.pi


class Interval:

    def __init__(self, lower_bound, upper_bound):
        assert np.all(lower_bound <= upper_bound), "lower bound has to be smaller or equal to upper bound"
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __neg__(self):
        return Interval(-self.upper_bound, -self.lower_bound)

    def __add__(self, other):
        if isinstance(other, Interval):
            return Interval(self.lower_bound + other.lower_bound,
                            self.upper_bound + other.upper_bound)
        else:
            return Interval(self.lower_bound + other, self.upper_bound + other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Interval):
            return Interval(self.lower_bound - other.upper_bound,
                            self.upper_bound - other.lower_bound)
        else:
            return Interval(self.lower_bound - other, self.upper_bound - other)

    def __rsub__(self, other):
        return Interval(other - self.upper_bound, other - self.lower_bound)

    def __mul__(self, other):
        if isinstance(other, Interval):
            return Interval(
                np.min([
                    self.lower_bound * other.lower_bound,
                    self.lower_bound * other.upper_bound,
                    self.upper_bound * other.lower_bound,
                    self.upper_bound * other.upper_bound
                ], axis=0),
                np.max([
                    self.lower_bound * other.lower_bound,
                    self.lower_bound * other.upper_bound,
                    self.upper_bound * other.lower_bound,
                    self.upper_bound * other.upper_bound
                ], axis=0)
            )
        else:
            return Interval(
                np.minimum(self.lower_bound * other, self.upper_bound * other),
                np.maximum(self.lower_bound * other, self.upper_bound * other)
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __eq__(self, other):
        if isinstance(other, Interval):
            return self.lower_bound == other.lower_bound and self.upper_bound == other.upper_bound
        return False

    def __repr__(self):
        return f"Interval(lower_bound={self.lower_bound}, upper_bound={self.upper_bound})"

    def center(self) -> Union[float, np.ndarray]:
        return (self.lower_bound + self.upper_bound) / 2

    def __getitem__(self, item):
        return Interval(self.lower_bound.__getitem__(item), self.upper_bound.__getitem__(item))

    def __len__(self):
        return self.lower_bound.__len__()

    # Make sure __radd__ etc work correctly with numpy ndarray
    __numpy_ufunc__ = None  # Numpy up to 13.0
    __array_ufunc__ = None  # Numpy 13.0 and above


def square(x: [Interval, float, np.ndarray]):
    if isinstance(x, Interval):
        # Default case with positive and negative ranges
        lower_bound = np.zeros_like(x.lower_bound)
        upper_bound = np.maximum(np.square(x.lower_bound), np.square(x.upper_bound))
        # Special case where all values are positive
        lower_bound = np.where(
            x.lower_bound >= 0,
            np.square(x.lower_bound),
            lower_bound
        )
        upper_bound = np.where(
            x.lower_bound >= 0,
            np.square(x.upper_bound),
            upper_bound
        )
        # Special case where all values are negative
        lower_bound = np.where(
            x.upper_bound <= 0,
            np.square(x.upper_bound),
            lower_bound
        )
        upper_bound = np.where(
            x.upper_bound <= 0,
            np.square(x.lower_bound),
            upper_bound
        )
        return Interval(lower_bound, upper_bound)
    else:
        return np.square(x)


def cos(theta: [Interval, float, np.ndarray]):
    return sin(theta + PI_HALF)


def sin(theta: [Interval, float, np.ndarray]):
    if isinstance(theta, Interval):
        offset = np.floor(theta.lower_bound / TWO_PI) * TWO_PI
        theta_lower = theta.lower_bound - offset
        theta_upper = theta.upper_bound - offset
        lower = np.minimum(np.sin(theta_lower), np.sin(theta_upper))
        upper = np.maximum(np.sin(theta_lower), np.sin(theta_upper))
        lower = np.where(
            np.logical_and(theta_lower <= 3 * PI_HALF, 3 * PI_HALF <= theta_upper),
            -1, lower
        )
        upper = np.where(
            np.logical_and(theta_lower <= PI_HALF, PI_HALF <= theta_upper),
            1, upper
        )
        lower = np.where(
            np.logical_and(theta_lower <= 7 * PI_HALF, 7 * PI_HALF <= theta_upper),
            -1, lower
        )
        upper = np.where(
            np.logical_and(theta_lower <= 5 * PI_HALF, 5 * PI_HALF <= theta_upper),
            1, upper
        )
        return Interval(lower, upper)
    else:
        return np.sin(theta)


def stack(elements: Iterable[Union[Interval, np.ndarray]], axis: int, convert=False):
    if convert or all([isinstance(it, Interval) for it in elements]):
        return Interval(
            lower_bound=np.stack([as_interval(element).lower_bound for element in elements], axis),
            upper_bound=np.stack([as_interval(element).upper_bound for element in elements], axis)
        )
    else:
        return np.stack(elements, axis)


def as_interval(element: Union[np.ndarray, float, Interval]) -> Interval:
    if isinstance(element, Interval):
        return element
    else:
        return Interval(element, element)


def zeros_like(element) -> np.ndarray:
    if isinstance(element, Interval):
        return np.zeros_like(element.lower_bound)
    else:
        return np.zeros_like(element)


def ones_like(element) -> np.ndarray:
    if isinstance(element, Interval):
        return np.ones_like(element.lower_bound)
    else:
        return np.ones_like(element)


def encode_rotation_box2d(points: np.ndarray, alpha: Interval, beta: Interval) -> Interval:
    sin_a = sin(alpha)
    cos_a = cos(alpha)
    sin_b = sin(beta)
    cos_b = cos(beta)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    x_rotated = (cos_a * x) - (sin_a * cos_b * y) + (sin_a * sin_b * z)
    y_rotated = (sin_a * x) + (cos_a * cos_b * y) - (cos_a * sin_b * z)
    z_rotated = (sin_b * y) + (cos_b * z)
    return stack([x_rotated, y_rotated, z_rotated], axis=1)


def encode_rotation_box3d(points: np.ndarray, alpha: Interval, beta: Interval, gamma: Interval) -> Interval:
    sin_a = sin(alpha)
    cos_a = cos(alpha)
    sin_b = sin(beta)
    cos_b = cos(beta)
    sin_c = sin(gamma)
    cos_c = cos(gamma)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    x_rotated = (cos_a * cos_b) * x + \
                (cos_a * sin_b * sin_c - sin_a * cos_c) * y + \
                (cos_a * sin_b * cos_c + sin_a * sin_c) * z
    y_rotated = (sin_a * cos_b) * x + \
                (sin_a * sin_b * sin_c + cos_a * cos_c) * y + \
                (sin_a * sin_b * cos_c - cos_a * sin_c) * z
    z_rotated = (-sin_b) * x + (cos_b * sin_c) * y + (cos_b * cos_c) * z
    return stack([x_rotated, y_rotated, z_rotated], axis=1)


if __name__ == '__main__':
    for i in range(1000):
        center = np.random.uniform(-100, 100, (2,))
        width = np.random.uniform(0, 2 * np.pi, (2,))
        alpha = Interval(center[0] - width[0], center[0] + width[0])
        beta = Interval(center[1] - width[1], center[1] + width[1])
        points = np.random.uniform(-1, 1, (1000, 3))
        encoding = encode_rotation_box2d(points, alpha, beta)
        for j in range(100):
            alpha0 = np.random.uniform(alpha.lower_bound, alpha.upper_bound)
            beta0 = np.random.uniform(beta.lower_bound, beta.upper_bound)
            x = points[:, 0]
            y = points[:, 1]
            z = points[:, 2]
            x_rotated = x * (np.cos(alpha0)) + y * (-np.sin(alpha0) * np.cos(beta0)) + z * (
                    np.sin(alpha0) * np.sin(beta0))
            y_rotated = x * (np.sin(alpha0)) + y * (np.cos(alpha0) * np.cos(beta0)) + z * (
                    -np.cos(alpha0) * np.sin(beta0))
            z_rotated = y * np.sin(beta0) + z * np.cos(beta0)
            rotated_points = np.stack([x_rotated, y_rotated, z_rotated], axis=1)
            assert np.all(encoding.lower_bound - 0.0001 <= rotated_points)
            assert np.all(encoding.upper_bound + 0.0001 >= rotated_points)
    print("2D box relaxation passed")

    for i in range(1000):
        center = np.random.uniform(-100, 100, (3,))
        width = np.random.uniform(0, 2 * np.pi, (3,))
        alpha = Interval(center[0] - width[0], center[0] + width[0])
        beta = Interval(center[1] - width[1], center[1] + width[1])
        gamma = Interval(center[2] - width[2], center[2] + width[2])
        points = np.random.uniform(-1, 1, (1000, 3))
        encoding = encode_rotation_box3d(points, alpha, beta, gamma)
        for j in range(100):
            alpha0 = np.random.uniform(alpha.lower_bound, alpha.upper_bound)
            beta0 = np.random.uniform(beta.lower_bound, beta.upper_bound)
            gamma0 = np.random.uniform(gamma.lower_bound, gamma.upper_bound)
            x = points[:, 0]
            y = points[:, 1]
            z = points[:, 2]
            x_rotated = x * (np.cos(alpha0) * np.cos(beta0)) + y * (
                    np.cos(alpha0) * np.sin(beta0) * np.sin(gamma0) - np.sin(alpha0) * np.cos(gamma0)) + z * (
                                np.cos(alpha0) * np.sin(beta0) * np.cos(gamma0) + np.sin(alpha0) * np.sin(gamma0))
            y_rotated = x * (np.sin(alpha0) * np.cos(beta0)) + y * (
                    np.sin(alpha0) * np.sin(beta0) * np.sin(gamma0) + np.cos(alpha0) * np.cos(gamma0)) + z * (
                                np.sin(alpha0) * np.sin(beta0) * np.cos(gamma0) - np.cos(alpha0) * np.sin(gamma0))
            z_rotated = x * (-np.sin(beta0)) + y * (np.cos(beta0) * np.sin(gamma0)) + z * (
                    np.cos(beta0) * np.cos(gamma0))
            rotated_points = np.stack([x_rotated, y_rotated, z_rotated], axis=1)
            assert np.all(encoding.lower_bound - 0.0001 <= rotated_points)
            assert np.all(encoding.upper_bound + 0.0001 >= rotated_points)
    print("3D box relaxation passed")
    print("Done.")
