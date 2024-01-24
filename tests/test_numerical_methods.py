import unittest

import numpy as np

from numerical_methods import first_derivative_cd, first_derivative_cd_dirichlet_bc


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
