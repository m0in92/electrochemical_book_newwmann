import numpy as np


class Grid1DFD:
    def __init__(self, x_begin: float, x_end: float, n_x: int):
        self.x_begin = x_begin
        self.x_end = x_end
        self.n_x_: int = n_x

    @property
    def dx(self) -> float:
        return self.x[1] - self.x[0]

    @property
    def x(self) -> np.ndarray:
        return np.linspace(self.x_begin, self.x_end, self.n_x_+1)

    @property
    def n_x(self) -> int:
        return self.n_x_

    def __str__(self):
        return str(self.x)

