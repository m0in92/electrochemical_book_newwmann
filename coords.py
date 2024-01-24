import numpy as np


class Grid1DFD:
    def __init__(self, x_begin: float, x_end: float, dx: float):
        self.x = np.arange(x_begin, x_end + dx, dx)

print(Grid1DFD(0,1,0.25).x)
