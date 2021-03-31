import itertools

import numpy as np

from relaxations.interval import Interval


def split_implicitly(bounds, intervals, encode_rotation, params):
    split_inc = [(param.upper_bound - param.lower_bound) / intervals for param in params]

    refined_lb = np.full_like(bounds.lower_bound, float('inf'))
    refined_ub = np.full_like(bounds.upper_bound, float('-inf'))

    for idx in itertools.product(range(intervals), repeat=len(params)):
        split_min = [param.lower_bound + idx[i] * split_inc[i] for i, param in enumerate(params)]
        split_max = [param.lower_bound + (idx[i] + 1) * split_inc[i] for i, param in enumerate(params)]
        split_intervals = [Interval(lb, ub) for lb, ub in zip(split_min, split_max)]
        candidate_bounds = encode_rotation(split_intervals)
        np.minimum(refined_lb, candidate_bounds.lower_bound, out=refined_lb)
        np.maximum(refined_ub, candidate_bounds.upper_bound, out=refined_ub)

    assert np.logical_or(
        bounds.lower_bound <= refined_lb,
        np.isclose(bounds.lower_bound, refined_lb)
    ).all()
    assert np.logical_or(
        refined_ub <= bounds.upper_bound,
        np.isclose(refined_ub, bounds.upper_bound)
    ).all()
    assert (refined_lb <= refined_ub).all()

    return Interval(refined_lb, refined_ub)
