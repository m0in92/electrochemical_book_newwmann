import unittest

import numpy as np

from numerical_methods import first_derivative_cd, first_derivative_cd_dirichlet_bc, tridiagonal_matrix_det, \
    TridiagonalMatrix


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


class TestTridiagonalMatrixClass(unittest.TestCase):
    a: np.ndarray = np.array([-2, -2, -2, -2, -2])
    b: np.ndarray = np.array([1, 1, 1, 1])
    c: np.ndarray = np.array([-1, -1, -1, -1])
    tri_matrix: TridiagonalMatrix = TridiagonalMatrix(a=a, b=b, c=c)

    def test_constructor(self):
        self.assertTrue(np.array_equal(self.a, self.tri_matrix.a))
        self.assertTrue(np.array_equal(self.b, self.tri_matrix.b))
        self.assertTrue(np.array_equal(self.c, self.tri_matrix.c))
        self.assertEqual(len(self.a), self.tri_matrix.n)

    def test_property_full_matrix(self):
        answer: np.ndarray = np.array([[-2, 1, 0, 0, 0],
                                       [-1, -2, 1, 0, 0],
                                       [0, -1, -2, 1, 0],
                                       [0, 0, -1, -2, 1],
                                       [0, 0, 0, -1, -2]])
        self.assertTrue(np.array_equal(answer, self.tri_matrix.full_matrix))

    def test_property_det(self):
        answer: float = -6
        self.assertTrue(answer, self.tri_matrix.det)

    def test_property_LU_matrix(self):
        a: np.ndarray = np.array([1, 7, 5])
        b: np.ndarray = np.array([1, 8])
        c: np.ndarray = np.array([2, 3])
        tridiag_matrix: TridiagonalMatrix = TridiagonalMatrix(a=a, b=b, c=c)

        answer_l: np.ndarray = np.array([[1, 0, 0], [2, 1, 0], [0, 0.6, 1]])
        answer_u: np.ndarray = np.array([[1, 1, 0], [0, 5, 8], [0, 0, 0.2]])

        self.assertTrue(np.array_equal(answer_l, tridiag_matrix.LU_matrix[0]))
        self.assertTrue(np.all(np.isclose(answer_u, tridiag_matrix.LU_matrix[1])))
    def test_method_thomas_alg_y(self):
        a: np.ndarray = np.array([1, 7, 5])
        b: np.ndarray = np.array([1, 8])
        c: np.ndarray = np.array([2, 3])
        d: np.ndarray = np.array([6, 9, 6])
        tridiag_matrix: TridiagonalMatrix = TridiagonalMatrix(a=a, b=b, c=c)

        answer: np.ndarray = np.array([6, -3, 7.8])

        self.assertTrue(np.all(np.isclose(tridiag_matrix.thomas_alg_y(d=d)[2], answer)))

    def test_method_thomas_alg(self):
        a: np.ndarray = np.array([1, 7, 5])
        b: np.ndarray = np.array([1, 8])
        c: np.ndarray = np.array([2, 3])
        d: np.ndarray = np.array([6, 9, 6])
        tridiag_matrix: TridiagonalMatrix = TridiagonalMatrix(a=a, b=b, c=c)

        answer: np.ndarray = np.array([69, -63, 39])

        self.assertTrue(np.allclose(answer, tridiag_matrix.thomas_alg(d=d)))



