import unittest

import numpy as np

from numerical_methods import first_derivative_cd, first_derivative_cd_dirichlet_bc, tridiagonal_matrix_det


class TestFirstDerivative(unittest.TestCase):
    def test_centered_difference(self):
        dx: float = 1.0
        y_prev: float = 0.0
        y_next: float = 1.0
        self.assertEqual(0.5, first_derivative_cd(y_next=y_next, y_prev=y_prev, dx=dx))

    def test_centered_difference_matrix_dirchilet_bc(self):
        n: int = 6
        final_result: np.ndarray = np.array([[1, 0, 0, 0, 0, 0],
                                             [-1, 0, 1, 0, 0, 0],
                                             [0, -1, 0, 1, 0, 0],
                                             [0, 0, -1, 0, 1, 0],
                                             [0, 0, 0, -1, 0, 1],
                                             [0, 0, 0, 0, 0, 1]])
        self.assertTrue(np.array_equal(final_result, first_derivative_cd_dirichlet_bc(n)))

    class TestTridiagonalMatrix(unittest.TestCase):
        def test_determinant(self):
            a: np.ndarray = np.array([-2, -2, -2, -2, -2])
            b: np.ndarray = np.array([1, 1, 1, 1])
            c: np.ndarray = np.array([1, 1, 1, 1])
            det: float = tridiagonal_matrix_det(a=a, b=b, c=c)

            self.assertEqual(-6, det)
