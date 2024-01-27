"""
Numerical solution of 1D diffusion equation using Crank-Nickelson Equation
"""

import numpy as np
import matplotlib.pyplot as plt

from initial_conditions import linear, plug, guassian  # functions representing the initial conditions
from numerical_methods import TridiagonalMatrix


# Problem parameters
a: float = 0.01
length: float = 1.0
total_sim_time: float = 5.0  # in s

# Simulation parameters
n_x: int = 100
x: np.ndarray = np.linspace(0, length, n_x+1)
dx: float = x[1] - x[0]

f: float = 0.16025649999999997

dt: float = f*dx**2/a
n_t: int = int(round(total_sim_time/dt))  # total number of simulation time steps
t: np.ndarray = np.linspace(0, total_sim_time, n_t+1)

# simulation related variables
c: np.ndarray = np.zeros((n_t+1, n_x+1))
c_prev: np.ndarray = np.zeros(n_x+1)

# Matrix and the column vector
A: TridiagonalMatrix = TridiagonalMatrix.toeplitz_matrix(a=f+1, b=-f/2, c=-f/2, n=n_x+1)
A.apply_dirchilet_bc_begin()
A.apply_dirichlet_bc_end()

b: np.ndarray = np.zeros(n_x+1)

# initial condition setup
for i in range(0, n_x+1):
    c_prev[i] = guassian(x=x[i])
c[0, :] = c_prev

# Simulation iteration
for n in range(0, n_t):
    for i in range(1, n_x):
        b[i] = c_prev[i] * (1-f) + (f/2) * c_prev[i-1] + (f/2) * c_prev[i+1]
    b[0] = b[n_x] = 0
    c[n+1, :] = A.thomas_alg(d=b)

    # update the variables
    c_prev = c[n+1, :]

for n in range(0, n_t+1, 500):
    print(n*dt)
    plt.plot(x, c[n], label=f't={n*dt:.2f}s')
plt.legend()
plt.show()
