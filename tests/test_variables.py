import unittest
from typing import Union

import numpy as np

from coords import Grid1DFD
from variables import Conc


def guassian(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Gaussian profile as initial condition."""
    return np.exp(-0.5 * ((x - 1.0 / 2.0) ** 2) / 0.05 ** 2)


class TestConc(unittest.TestCase):
    length: float = 1.0
    n_x: int = 5
    x_begin: float = 0.0
    x_end: float = length
    x: Grid1DFD = Grid1DFD(x_begin=x_begin, x_end=x_end, n_x=n_x)
    n_t: int = 3120

    def test_constructor(self):
        c: Conc = Conc(grid=self.x, func_init=guassian, n_t=self.n_t)

        self.assertAlmostEqual(c.init[0], 1.928749848e-22)
        self.assertAlmostEqual(c.init[1], 1.522997974e-8)
        self.assertAlmostEqual(c.init[2], 1.353352832e-1)
        self.assertAlmostEqual(c.init[3], 1.353352832e-1)
        self.assertAlmostEqual(c.init[4], 1.522997974e-8)
        self.assertAlmostEqual(c.init[5], 1.928749848e-22)

        self.assertAlmostEqual(c.values[0], 1.928749848e-22)
        self.assertAlmostEqual(c.init[1], 1.522997974e-8)
        self.assertAlmostEqual(c.values[2], 1.353352832e-1)
        self.assertAlmostEqual(c.values[3], 1.353352832e-1)
        self.assertAlmostEqual(c.values[4], 1.522997974e-8)
        self.assertAlmostEqual(c.values[5], 1.928749848e-22)

        self.assertEqual(self.n_t + 1, c.full_data.shape[0])
        self.assertEqual(8, c.full_data.shape[1])
        full_data_first_row: np.ndarray = np.array([0.0, 0.0,
                                                    1.928749848e-22, 1.522997974e-8,
                                                    1.353352832e-1, 1.353352832e-1,
                                                    1.522997974e-8, 1.928749848e-22])
        full_data_second_row: np.ndarray = np.array([0.0, 0.0,
                                                     0, 0,
                                                     0, 0,
                                                     0, 0])
        self.assertTrue(np.allclose(full_data_first_row, c.full_data[0, :]))
        for n in range(1, c.full_data.shape[0]):
            self.assertTrue(np.allclose(full_data_second_row, c.full_data[n, :]))

    def test_property_values(self):
        c: Conc = Conc(grid=self.x, func_init=guassian, n_t=self.n_t)
        c_values_actual: np.ndarray = np.array([1.928749848e-22, 1.522997974e-8,
                                                1.353352832e-1, 1.353352832e-1,
                                                1.522997974e-8, 1.928749848e-22])
        self.assertTrue(np.allclose(c_values_actual, c.values))
        c.values = np.zeros(self.n_x + 1)
        self.assertTrue(np.allclose(np.zeros(self.n_x + 1), c.values))
