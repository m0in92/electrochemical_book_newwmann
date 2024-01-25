from typing import Union, Callable

import numpy as np
import matplotlib.pyplot as plt

from coords import Grid1DFD
from benchmarks.diffusion_equation_1D import solver_FE_simple
from variables import Conc


def func_c_init(a_coord: float) -> float:
    return a_coord


def func_c_init_plug(x: float) -> float:
    if abs(x - 1.0 / 2.0) > 0.1:
        return 0
    else:
        return 1


def guassian(x):
    """Gaussian profile as initial condition."""
    return np.exp(-0.5*((x-1.0/2.0)**2)/0.05**2)


a: float = 0.01  # unit-less
# dt: float = 0.00641026  # in s
n_x: int = 100
total_sim_time: float = 10.0  # in s
length: float = 1.0

grid: Grid1DFD = Grid1DFD(x_begin=0.0, x_end=length, n_x=n_x)
conc = Conc(grid=grid, func_c_init=guassian)

f: float = 0.16025649999999997
dt: float = f*grid.dx**2/a
n_t: int = int(round(total_sim_time/dt))  # total number of simulation time steps

c = np.zeros(n_x+1)
c_prev = conc.c_init

print("---------------------------------------------")
print("SIMULATION PARAMETERS")
print("c_init: ", conc.c_init)
print("x: ", grid.x)
print("dx: ", grid.dx)
print("dt:", dt)
print("F", f)
print("---------------------------------------------")

for n_ in range(0, n_t):
    # compute the concentration at all the grid points
    for i_ in range(1, n_x):
        c[i_] = c_prev[i_] + f*(c_prev[i_-1] - 2*c_prev[i_] + c_prev[i_+1])

    # Insert the boundary conditions
    c[0] = 0.0
    c[-1] = 0.0

    # update the concentrations of the previous time arrays for the next iterations
    c_prev, c = c, c_prev
    conc.values = c

benchmark_results = solver_FE_simple(I=guassian, a=a, L=length,
                                     Nx=n_x, F=f, T=total_sim_time)
#
# print(benchmark_results)
print("bench mark solution time in s: ", benchmark_results[-1])

plt.scatter(grid.x, conc.values)
plt.plot(grid.x, conc.values)
plt.scatter(benchmark_results[1], benchmark_results[0])
plt.plot(benchmark_results[1], benchmark_results[0])
plt.show()
