from typing import Union, Callable

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

from coords import Grid1DFD
from benchmarks.diffusion_equation_1D import solver_BE_simple
from variables import Conc
from initial_conditions import linear, plug, guassian


a: float = 0.01  # unit-less
# dt: float = 0.00641026  # in s
n_x: int = 100
length: float = 1.0

x: np.ndarray = np.linspace(0, length, n_x+1)
dx: float = x[1] - x[0]

f: float = 0.16025649999999997

total_sim_time: float = 1.0  # in s
dt: float = f*dx**2/a
n_t: int = int(round(total_sim_time/dt))  # total number of simulation time steps
t: np.ndarray = np.linspace(0, total_sim_time, n_t+1)

c: np.ndarray = np.zeros(n_x+1)
c_prev: np.ndarray = np.zeros(n_x+1)

# Setup data structures for the linear system
A = np.zeros((n_x+1, n_x+1))
b = np.zeros(n_x+1)

for i in range(1, n_x):
    A[i, i-1] = -f
    A[i, i+1] = -f
    A[i, i] = 1+2*f
A[0, 0] = A[n_x, n_x] = 1

# Setup initial conditions
for i in range(0, n_x+1):
    c_prev[i] = guassian(x=x[i])

for n in range(0, n_t):
    for i in range(1, n_x):
        b[i] = c_prev[i]
    b[0] = b[n_x] = 0
    c[:] = scipy.linalg.solve(A, b)

    c_prev = c

# benchmark results
c_benchmark, x_benchmark, t_benchmark, sol_time_benchmark = solver_BE_simple(I=guassian,
                                                                             a=a, L=length, Nx=n_x,
                                                                             F=f, T=total_sim_time)

plt.plot(x, c)
plt.plot(x_benchmark, c_benchmark, label="benchmark")

plt.legend()
plt.show()
