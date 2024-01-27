"""
This module solves for the potential profile of the positive electrode.
"""
import time

import numpy as np
import matplotlib.pyplot as plt

from numerical_methods import TridiagonalMatrix


a: float = 885  # in 1/m
F: float = 96487  # in C/mol
j: float = 13.5/F  # in A/m2
sigma: float = 100  # in S/m
epsilon: float = 0.485  # unit-less
brug: float = 1.5  # unit-less

sigma_eff: float = sigma * (epsilon)**brug

electrode_thickness: float = 8.e-5  # in m
n_x: int = 100
x: np.ndarray = np.linspace(0, electrode_thickness, n_x)
dx: float = x[1] - x[0]
x = np.append(x, electrode_thickness+dx)
print(len(x))

upper_diagonal: np.ndarray = -1 * np.ones(n_x)
main_diagonal: np.ndarray = 2 * np.ones(n_x+1)
lower_diagonal = -1 * np.ones(n_x)
M: TridiagonalMatrix = TridiagonalMatrix.toeplitz_matrix(a=2, b=-1, c=-1, n=n_x+1)
M_: np.ndarray = np.diag(main_diagonal) + np.diag(upper_diagonal, 1) + np.diag(lower_diagonal, -1)


M_[0, 0] = -1
M_[0, 1] = 1
# M_[0, 1] = 0
M_[-1, -1] = -3
M_[-1, -2] = 1
# M_[-1, -3] = -1
print(M_)

M.apply_dirchilet_bc_begin()
M.apply_newman_bc_end()

b: np.ndarray = np.zeros(n_x+1).reshape(-1, 1)
b[0] = 0
b[-1] = a*F*j*(dx**2)/sigma_eff - 2 * (4)

for i in range(1, n_x):
    b[i] = a*F*j*(dx**2)/sigma_eff

t_start_: time.time = time.time()
phi_: np.ndarray = np.linalg.inv(M_)@b
t_end_: time.time = time.time()

t_start: time.time = time.time()
phi: np.ndarray = M.thomas_alg(d=b)
t_end: time.time = time.time()

print("numpy: ", t_end_-t_start_)
print("mine: ", t_end-t_start)
print(phi_)

plt.plot(x, phi_)
plt.ylim(3, 4.2)
# plt.plot(x, phi)
# plt.vlines(electrode_thickness, ymin=phi[-1], ymax=0.0, colors='r')
plt.show()



