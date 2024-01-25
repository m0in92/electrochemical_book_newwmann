import unittest

import numpy as np

from coords import Grid1DFD


class TestGrid1dFD(unittest.TestCase):
    def test_x(self):
        answer: np.ndarray = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        grid: Grid1DFD = Grid1DFD(x_begin=0.0, x_end=1.0, dx=0.25)

        self.assertTrue(np.allclose(answer, grid.x))
