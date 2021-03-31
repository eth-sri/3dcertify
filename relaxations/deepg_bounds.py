from pathlib import Path
from typing import List, Tuple

import numpy as np

from relaxations.interval import Interval
from relaxations.linear_bounds import LinearBounds


def load_spec(spec_dir: Path, counter: int) -> List[Tuple[List[Interval], Interval, LinearBounds]]:
    parameters = list()
    interval_bounds = list()
    lower_biases = list()
    upper_biases = list()
    lower_weights = list()
    upper_weights = list()

    with (spec_dir / f'{counter}.csv').open('r') as f:
        split_parameters = list()
        split_interval_bounds = list()
        split_lower_biases = list()
        split_upper_biases = list()
        split_lower_weights = list()
        split_upper_weights = list()

        for line in f.readlines():
            if '|' in line:
                lower, upper = line.strip().split(' | ')
                lower = [float(v) for v in lower.split(' ')]
                upper = [float(v) for v in upper.split(' ')]

                split_lower_biases.append(lower[0])
                split_upper_biases.append(upper[0])
                split_lower_weights.append(lower[1:])
                split_upper_weights.append(upper[1:])

            elif 'SPEC_FINISHED' in line:
                parameters.append(np.asarray(split_parameters))
                interval_bounds.append(np.asarray(split_interval_bounds))
                lower_biases.append(np.asarray(split_lower_biases))
                upper_biases.append(np.asarray(split_upper_biases))
                lower_weights.append(np.asarray(split_lower_weights))
                upper_weights.append(np.asarray(split_upper_weights))

                split_parameters = list()
                split_interval_bounds = list()
                split_lower_biases = list()
                split_upper_biases = list()
                split_lower_weights = list()
                split_upper_weights = list()

            elif line.startswith('('):
                split_interval_bounds.extend(eval(line))

            else:
                split_parameters.append([float(v) for v in line.strip().split(' ')])

    parameters = np.array(parameters)
    interval_bounds = np.asarray(interval_bounds)
    lower_biases = np.asarray(lower_biases)
    upper_biases = np.asarray(upper_biases)
    lower_weights = np.asarray(lower_weights)
    upper_weights = np.asarray(upper_weights)

    result = list()

    for i in range(len(parameters)):
        params = [Interval(param[0], param[1]) for param in parameters[i]]

        bounds = Interval(
            lower_bound=interval_bounds[i][:, 0],
            upper_bound=interval_bounds[i][:, 1]
        )

        constraints = LinearBounds(
            upper_slope=upper_weights[i],
            upper_offset=upper_biases[i],
            lower_slope=lower_weights[i],
            lower_offset=lower_biases[i]
        )
        result.append((params, bounds, constraints))

    return result
