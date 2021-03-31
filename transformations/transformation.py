from typing import Union, List

import numpy as np

from relaxations.interval import Interval


class Transformation:

    def __init__(self, num_params):
        self.num_params = num_params

    def transform(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[np.ndarray, Interval]:
        raise NotImplementedError()

    def gradient_params(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[np.ndarray], List[Interval]]:
        raise NotImplementedError()

    def gradient_points(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[np.ndarray], List[Interval]]:
        raise NotImplementedError()

    def hessian_params(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[List[np.ndarray]], List[List[Interval]]]:
        raise NotImplementedError()

    def hessian_points(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[List[np.ndarray]], List[List[Interval]]]:
        raise NotImplementedError()

    def hessian_points_params(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[List[np.ndarray]], List[List[Interval]]]:
        raise NotImplementedError()
