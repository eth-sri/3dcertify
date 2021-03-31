from typing import Union, List

import numpy as np

import relaxations.interval as iv
from relaxations.interval import Interval
from transformations.transformation import Transformation

NUM_COORDINATES = 3


class Composition(Transformation):

    def __init__(self, outer: Transformation, inner: Transformation):
        super().__init__(outer.num_params + inner.num_params)
        self.outer = outer
        self.inner = inner

    def transform(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[np.ndarray, Interval]:
        assert len(params) == self.num_params
        return self.outer.transform(self.inner.transform(points, params[self.outer.num_params:]), params[:self.outer.num_params])

    def gradient_params(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[np.ndarray], List[Interval]]:
        params_outer = params[:self.outer.num_params]
        params_inner = params[self.outer.num_params:]

        transformed_inner = self.inner.transform(points, params_inner)

        # the partial derivative of the composition wrt. the parameters of the outer function is simply
        # that of the outer function since the inner function does not depend on those parameters
        result = self.outer.gradient_params(transformed_inner, params_outer)

        # for the partial derivative wrt. the parameters of the inner function, we apply the chain rule
        # (dx, dy, dz) x (num_points) x (x, y, z)
        d_outer_d_points = self.outer.gradient_points(transformed_inner, params_outer)
        # (d_theta_i) x (num_points) x (x, y, z)
        d_inner_d_params = self.inner.gradient_params(points, params_inner)

        for inner_derivative in d_inner_d_params:
            derivative = iv.zeros_like(points)
            for i in range(NUM_COORDINATES):
                derivative = derivative + d_outer_d_points[i] * inner_derivative[:, i, np.newaxis]
            result.append(derivative)

        return result

    def gradient_points(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[np.ndarray], List[Interval]]:
        params_outer = params[:self.outer.num_params]
        params_inner = params[self.outer.num_params:]

        transformed_inner = self.inner.transform(points, params_inner)

        # All parameters are constant - simply apply chain rule on point parameters
        # (dx, dy, dz) x (num_points) x (x, y, z)
        d_outer_d_points = self.outer.gradient_points(transformed_inner, params_outer)
        # (dx, dy, dz) x (num_points) x (x, y, z)
        d_inner_d_params = self.inner.gradient_points(points, params_inner)
        result = []
        for inner_derivative in d_inner_d_params:
            derivative = iv.zeros_like(points)
            for i in range(NUM_COORDINATES):
                derivative = derivative + d_outer_d_points[i] * inner_derivative[:, i, np.newaxis]
            result.append(derivative)

        return result

    def hessian_params(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[List[np.ndarray]], List[List[Interval]]]:
        params_outer = params[:self.outer.num_params]
        params_inner = params[self.outer.num_params:]

        result: Union[List[List[np.ndarray]], List[List[Interval]]] = [[None] * self.num_params for _ in range(self.num_params)]

        transformed_inner = self.inner.transform(points, params_inner)
        # (d_theta_i x d_theta_j x num_points x coords)
        outer_gradient_points = self.outer.gradient_points(transformed_inner, params_outer)
        outer_hessian_params = self.outer.hessian_params(transformed_inner, params_outer)
        outer_hessian_points = self.outer.hessian_points(transformed_inner, params_outer)
        outer_hessian_points_params = self.outer.hessian_points_params(transformed_inner, params_outer)
        inner_gradient_params = self.inner.gradient_params(points, params_inner)
        inner_gradient_points = self.inner.gradient_points(points, params_inner)
        inner_hessian_params = self.inner.hessian_params(points, params_inner)

        # upper left quadrant: partial derivatives wrt. outer parameters only
        for i in range(self.outer.num_params):
            for j in range(self.outer.num_params):
                result[i][j] = outer_hessian_params[i][j]

        # lower right quadrant: partial derivatives wrt. inner parameters only
        for i in range(self.inner.num_params):
            for j in range(self.inner.num_params):
                derivative = iv.zeros_like(points)
                for k in range(NUM_COORDINATES):
                    derivative = derivative + outer_gradient_points[k] * inner_hessian_params[i][j][:, k, np.newaxis]
                for k in range(NUM_COORDINATES):
                    for l in range(NUM_COORDINATES):
                        derivative = derivative + outer_hessian_points[k][l] * inner_gradient_points[i][:, k, np.newaxis] * inner_gradient_points[j][:, l, np.newaxis]
                result[self.outer.num_params + i][self.outer.num_params + j] = derivative

        # upper right and lower left quadrants: partial derivative wrt. inner and outer parameters
        for i in range(self.outer.num_params):
            for j in range(self.inner.num_params):
                derivative = iv.zeros_like(points)
                for k in range(NUM_COORDINATES):
                    derivative = derivative + outer_hessian_points_params[k][i] * inner_gradient_params[j][:, k, np.newaxis]
                result[i][self.outer.num_params + j] = derivative
                result[self.outer.num_params + j][i] = derivative

        return result

    def hessian_points(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[List[np.ndarray]], List[List[Interval]]]:
        params_outer = params[:self.outer.num_params]
        params_inner = params[self.outer.num_params:]

        inner_transformed = self.inner.transform(points, params_inner)
        outer_gradient_points = self.outer.gradient_points(inner_transformed, params_outer)
        inner_hessian_points = self.inner.hessian_points(points, params_inner)
        outer_hessian_points = self.outer.hessian_points(inner_transformed, params_outer)
        inner_gradient_points = self.inner.gradient_points(points, params_inner)

        # noinspection PyTypeChecker
        result: Union[List[List[np.ndarray]], List[List[Interval]]] = [[None] * NUM_COORDINATES for _ in range(NUM_COORDINATES)]
        for i in range(NUM_COORDINATES):
            for j in range(NUM_COORDINATES):
                derivative = iv.zeros_like(points)
                for k in range(NUM_COORDINATES):
                    derivative = derivative + outer_gradient_points[k] * inner_hessian_points[i][j][:, k, np.newaxis]
                for k in range(NUM_COORDINATES):
                    for l in range(NUM_COORDINATES):
                        derivative = derivative + outer_hessian_points[k][l] * inner_gradient_points[i][:, k, np.newaxis] * inner_gradient_points[j][:, l, np.newaxis]
                result[i][j] = derivative
        return result

    def hessian_points_params(self, points: np.ndarray, params: Union[List[float], List[Interval]]) -> Union[List[List[np.ndarray]], List[List[Interval]]]:
        params_outer = params[:self.outer.num_params]
        params_inner = params[self.outer.num_params:]

        inner_transformed = self.inner.transform(points, params_inner)
        inner_gradient_params = self.inner.gradient_params(points, params_inner)
        inner_gradient_points = self.inner.gradient_points(points, params_inner)
        inner_hessian_points_params = self.inner.hessian_points_params(points, params_inner)

        outer_gradient_points = self.outer.gradient_points(inner_transformed, params_outer)
        outer_hessian_points = self.outer.hessian_points(inner_transformed, params_outer)
        outer_hessian_points_params = self.outer.hessian_points_params(inner_transformed, params_outer)

        # noinspection PyTypeChecker
        result: Union[List[List[np.ndarray]], List[List[Interval]]] = [[None] * self.num_params for _ in range(NUM_COORDINATES)]
        # partial derivatives wrt points and outer params
        for i in range(NUM_COORDINATES):
            for j in range(self.outer.num_params):
                derivative = iv.zeros_like(points)
                for k in range(NUM_COORDINATES):
                    derivative = derivative + outer_hessian_points_params[k][j] * inner_gradient_points[i][:, k, np.newaxis]
                result[i][j] = derivative

        for i in range(NUM_COORDINATES):
            for j in range(self.inner.num_params):
                derivative = iv.zeros_like(points)
                for k in range(NUM_COORDINATES):
                    derivative = derivative + outer_gradient_points[k] * inner_hessian_points_params[i][j][:, k, np.newaxis]
                for k in range(NUM_COORDINATES):
                    for l in range(NUM_COORDINATES):
                        derivative = derivative + outer_hessian_points[k][l] * inner_gradient_points[i][:, k, np.newaxis] * inner_gradient_params[j][:, l, np.newaxis]
                result[i][self.outer.num_params + j] = derivative
        return result
