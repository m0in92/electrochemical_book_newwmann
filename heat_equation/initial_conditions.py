from typing import Union

import numpy as np


def linear(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return x


def plug(x: float) -> float:
    if abs(x - 1.0 / 2.0) > 0.1:
        return 0
    else:
        return 1


def guassian(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Gaussian profile as initial condition."""
    return np.exp(-0.5*((x-1.0/2.0)**2)/0.05**2)
