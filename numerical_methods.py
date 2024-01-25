from typing import Self

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


class TridiagonalMatrix:
    def __init__(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> None:
        """
        Class constructor
        :param a: The vector containing the entries of the major diagonal
        :param b: The vector containing the entries of the upper diagonal.
        :param c: The vector containing the entries of the lower diagonal.
        :return: None
        """
        if (len(b) != len(a)-1) | len(c) != len(a)-1:
            raise ValueError("Length of vector b or c should be an order less than that of vector a")
        self.a = a
        self.b = b
        self.c = c
        self.n = len(a)

    @classmethod
    def toeplitz_matrix(cls, a: float, b: float, c: float, n: int) -> Self:
        if n < 2:
            raise ValueError("n, the size of the square matrix needs to be atleast 2 in size.,")
        a_: np.ndarray = a * np.ones(n)
        b_: np.ndarray = b * np.ones(n-1)
        c_: np.ndarray = c * np.ones(n-1)
        return cls(a=a_, b=b_, c=c_)

    @property
    def full_matrix(self) -> np.ndarray:
        """
        Returns a numpy array containing the full matrix
        :return:
        """
        return np.diag(self.a) + np.diag(self.b, 1) + np.diag(self.c, -1)

    @property
    def det(self) -> float:
        """
        This method solves for the determinant of a tri-diagonal matrix.
        :return: (float) matrix determinant
        """
        f_n: list = []
        for i in range(self.n):
            if i == 0:
                f_n.append(self.a[0])
            elif i == 1:
                f_n.append(self.a[i] * f_n[i - 1] - self.c[i - 1] * self.b[i - 1])
            else:
                f_n.append(self.a[i] * f_n[i - 1] - self.c[i - 1] * self.b[i - 1] * f_n[i - 2])
        return f_n[-1]

    @property
    def lu(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the L and U matrix from the LU decomposition
        :return: tuple with the first entry containing the L matrix and the second matrix containing the U matrix
        """
        # define the L and U vectors
        u: np.ndarray = np.zeros(self.n)
        l: np.ndarray = np.zeros(self.n - 1)

        # from thomas algorithm a_1 = u_1
        u[0] = self.a[0]

        # Now calculate l_i and u_i+1 until all the values have been calculated
        for i_ in range(self.n-1):
            l[i_] = self.c[i_] / u[i_]
            u[i_+1] = self.a[i_+1] - l[i_] * self.b[i_]

        return l, u

    @property
    def lu_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        l, u = self.lu
        return np.diag(np.ones(self.n)) + np.diag(l, -1), np.diag(u) + np.diag(self.b, 1)

    def thomas_alg_y(self, d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y: np.ndarray = np.zeros(self.n)
        y[0] = d[0]
        l, u = self.lu

        for i_ in range(1, self.n):
            y[i_] = d[i_] - l[i_-1]*y[i_-1]

        return l, u, y

    def thomas_alg(self, d: np.ndarray) -> np.ndarray:
        _, u, y = self.thomas_alg_y(d=d)
        x: np.ndarray = np.zeros(self.n)
        x[-1] = y[-1] / u[-1]

        for i_ in range(self.n-2, -1, -1):
            x[i_] = (y[i_] - self.b[i_] * x[i_+1])/u[i_]

        return x

    def apply_dirchilet_bc_begin(self) -> None:
        """
        Applies Dirichilet boundary conditions at the beginning of the matrix.
        :return: None
        """
        self.a[0] = 1.0
        self.b[0] = 0

    def apply_newman_bc_end(self) -> None:
        """
        Applies a Newman boundary condition at the end of the matrix
        :return: None
        """
        self.a[-1] = 1.0
        self.c[-1] = -1.0

    def __str__(self):
        return np.array_str(self.full_matrix)


