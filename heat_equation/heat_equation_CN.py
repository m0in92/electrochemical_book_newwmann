"""
Numerical solution of 1D diffusion equation using Crank-Nickelson Equation
"""

import numpy as np
import matplotlib.pyplot as plt

from coords import Grid1DFD
from initial_conditions import linear, plug, guassian  # functions representing the initial conditions
from numerical_methods import TridiagonalMatrix
from variables import Conc


# Problem parameters
a: float = 0.01
length: float = 1.0
total_sim_time: float = 5.0  # in s

# Simulation parameters
n_x: int = 100
x_begin: float = 0.0
x_end: float = length
x: Grid1DFD = Grid1DFD(x_begin=x_begin, x_end=x_end, n_x=n_x)

f: float = 0.16025649999999997

dt: float = f*x.dx**2/a
n_t: int = int(round(total_sim_time/dt))  # total number of simulation time steps
t: np.ndarray = np.linspace(0, total_sim_time, n_t+1)

# simulation related variables
c: Conc = Conc(grid=x, func_init=guassian, n_t=n_t)

# Matrix and the column vector
A: TridiagonalMatrix = TridiagonalMatrix.toeplitz_matrix(a=f+1, b=-f/2, c=-f/2, n=n_x+1)
A.apply_dirchilet_bc_begin()
A.apply_dirichlet_bc_end()

b: np.ndarray = np.zeros(n_x+1)

# Simulation iteration
for n in range(0, n_t):
    for i in range(1, n_x):
        b[i] = c.prev[i] * (1 - f) + (f / 2) * c.prev[i - 1] + (f / 2) * c.prev[i + 1]
    b[0] = b[n_x] = 0
    c_: np.ndarray = A.thomas_alg(d=b)

    # update the variables
    c.update_after_simulation_iteration(t=(n+1)*dt, n_t=n, new_values=c_)

for n in range(0, n_t+1, 500):
    plt.plot(x.x, c.full_data[n, 2:], label=f't={n*dt:.2f}s')
plt.legend()
plt.show()
