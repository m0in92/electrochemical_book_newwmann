import numpy as np
import numpy.typing as npt


def first_derivative_cd(y_next, y_prev, dx: float) -> float:
    return (y_next - y_prev) / (2 * dx)


def first_derivative_cd_dirichlet_bc(n: int) -> npt.ArrayLike:
    zeros: np.ndarray = np.zeros(n)
    zeros[0] = 1
    zeros[-1] = 1

    neg_ones: np.ndarray = -np.ones(n-1)
    neg_ones[-1] = 0

    ones: np.ndarray = np.ones(n-1)
    ones[0] = 0
    return np.diag(zeros) + np.diag(neg_ones, -1) + np.diag(ones, 1)

