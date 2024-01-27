from typing import Callable, Union

import numpy as np

from coords import Grid1DFD


class Conc:
    def __init__(self, grid: Grid1DFD, func_init: Callable, n_t: int):
        self.grid = grid
        self.init = self.create_init(func_init=func_init)
        self.values_: np.ndarray = self.init  # initial condition
        self.n_t: int = n_t

        self.full_data_: np.ndarray = np.zeros((self.n_t + 1, self.grid.n_x + 3))
        self.set_full_data(t=0.0, n_t=0, new_values=self.values)

        self.prev: np.ndarray = self.init

    def create_init(self, func_init: Callable) -> np.ndarray:
        init_: np.ndarray = np.zeros(len(self.grid.x))
        if isinstance(func_init, Callable):
            for i in range(0, self.grid.n_x + 1):
                init_[i] = func_init(self.grid.x[i])
        return init_

    @property
    def values(self) -> np.ndarray:
        return self.values_

    @values.setter
    def values(self, new_values: np.ndarray):
        self.values_ = new_values

    @property
    def full_data(self) -> np.ndarray:
        return self.full_data_

    def set_full_data(self, t: float, n_t: int, new_values: np.ndarray) -> None:
        self.full_data[n_t, 0] = t
        self.full_data[n_t, 2:] = new_values

    def update_after_simulation_iteration(self, t: float, n_t: int, new_values: np.ndarray) -> None:
        self.set_full_data(t=t, n_t=n_t, new_values=new_values)
        self.values = new_values
        self.prev = new_values

    def __str__(self):
        return str(self.values)

