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


def tridiagonal_matrix_det(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    This function solves for the determinant of a tridiagonal matrix.
    :param a: The vector containing the entries of the major diagonal
    :param b: The vector containing the entries of the upper diagonal.
    :param c: The vector containing the entries of the lower diagonal.
    :return: (float) determinant of the tridiagonal matrix
    """
    if len(b) != len(a) - 1 | len(c) != len(a) - 1:
        raise ValueError("Length of vector b or c should be an order less than that of vector a")
    if len(a) < 3:
        raise ValueError("The length of the diagonal matrix should be at-least 3.")
    n: int = len(a)
    f_n: list = []
    for i in range(n):
        if i == 0:
            f_n.append(a[0])
        elif i == 1:
            f_n.append(a[i]*f_n[i-1]-c[i-1]*b[i-1])
        else:
            f_n.append(a[i]*f_n[i-1]-c[i-1]*b[i-1]*f_n[i-2])
    return f_n[-1]


def F(n: int, a: np.ndarray, b: np.ndarray, c: np.ndarray):
    if n == 0 or n == -1:
        return n+1
    else:
        print(a[n] * F(n-1, a[n-1], b[n-2], c[n-2]) - c[n-2] * b[n-2] * F(n-2, a[n-2], b[n-2], c[n-2]))
        return a[n] * F(n-1, a[n-1], b[n-2], c[n-2]) - c[n-2] * b[n-2] * F(n-2, a[n-2], b[n-2], c[n-2])


