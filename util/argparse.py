from pathlib import Path

import numpy as np


def parse_theta(arg: str) -> float:
    return float(arg[:-3]) * np.pi / 180.0 if arg.endswith('deg') else float(arg)


def absolute_path(arg: str) -> Path:
    return Path(arg).absolute()
