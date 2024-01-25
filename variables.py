from typing import Callable, Union

import numpy as np

from coords import Grid1DFD


class Conc:
    def __init__(self, grid: Grid1DFD, func_c_init: Union[Callable, np.ndarray]):
        self.grid = grid
        if isinstance(func_c_init, Callable):
            self.c_init: np.ndarray = np.zeros(len(self.grid.x))
            for i in range(0, self.grid.n_x + 1):
                self.c_init[i] = func_c_init(grid.x[i])