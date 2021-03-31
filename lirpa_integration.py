import numpy as np
import torch
from auto_LiRPA.perturbations import Perturbation
from auto_LiRPA.utils import LinearBound

from relaxations import taylor


class SemanticTransformation(Perturbation):

    def __init__(self, transformation, params):
        super().__init__()
        self.transformation = transformation
        self.params = params

    def init(self, x, aux=None, forward=False):
        if forward:
            raise NotImplementedError()

        x_np = x.cpu().numpy()
        original_shape = x_np.shape
        x_np = np.reshape(x_np, (-1, original_shape[-1]))
        interval_bounds = self.transformation.transform(x_np, self.params)
        lb = np.reshape(interval_bounds.lower_bound, original_shape)
        ub = np.reshape(interval_bounds.upper_bound, original_shape)
        lb = torch.tensor(lb, device=x.device)
        ub = torch.tensor(ub, device=x.device)
        assert x.size() == lb.size() and x.size() == ub.size(), \
            f"bounds must have the same shape as x. Got x:{x.size()}, lb:{lb.size()}, ub:{ub.size()}"
        return LinearBound(None, None, None, None, lb, ub), x, None

    """Given an variable x and its bound matrix A, compute worst case bound according to semantic transformation."""

    def concretize(self, x, A, sign=-1, aux=None):
        n_batch = A.shape[0]
        n_outputs = A.shape[1]
        n_values = A.shape[2]
        n_points = x.shape[1]
        n_coords = x.shape[2]
        n_params = len(self.params)
        assert n_values == n_points * n_coords

        # Computing linear constraints based on taylor relaxations of transformation
        x_np = x.cpu().numpy()
        x_np = np.reshape(x_np, (n_batch * n_points, n_coords))
        bounds = taylor.encode(self.transformation, x_np, self.params)

        lower_offset = torch.tensor(bounds.lower_offset.reshape((n_batch, n_values, 1)), device=x.device)
        upper_offset = torch.tensor(bounds.upper_offset.reshape((n_batch, n_values, 1)), device=x.device)
        lower_slope = torch.tensor(bounds.lower_slope.reshape((n_batch, n_values, n_params)), device=x.device)
        upper_slope = torch.tensor(bounds.upper_slope.reshape((n_batch, n_values, n_params)), device=x.device)

        # Backwards propagate coefficients through linear relaxation of transformation
        if sign == -1:  # computing lower bound
            new_A = torch.matmul(A.clamp(min=0.0), lower_slope) + torch.matmul(A.clamp(max=0.0), upper_slope)
            offset = torch.matmul(A.clamp(min=0.0), lower_offset) + torch.matmul(A.clamp(max=0.0), upper_offset)
        elif sign == 1:  # computing upper bound
            new_A = torch.matmul(A.clamp(min=0.0), upper_slope) + torch.matmul(A.clamp(max=0.0), lower_slope)
            offset = torch.matmul(A.clamp(min=0.0), upper_offset) + torch.matmul(A.clamp(max=0.0), lower_offset)
        else:
            raise RuntimeError(f"Invalid sign value: {sign}")

        # Instantiate bounds based on valid parameter ranges. Same implementation as for L-inf perturbation in PerturbationLpNorm
        lb = torch.tensor([[p.lower_bound] for p in self.params], dtype=x.dtype, device=x.device).reshape((1, n_params, 1))
        ub = torch.tensor([[p.upper_bound] for p in self.params], dtype=x.dtype, device=x.device).reshape((1, n_params, 1))

        center = (ub + lb) / 2.0
        diff = (ub - lb) / 2.0

        bound = new_A.matmul(center) + sign * new_A.abs().matmul(diff)

        result = bound + offset
        assert result.shape == (n_batch, n_outputs, 1)
        return result.squeeze(-1)
