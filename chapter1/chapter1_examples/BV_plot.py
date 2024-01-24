import numpy as np
import matplotlib.pyplot as plt

from chapter1.common_equations import butler_volmer


eta_s_1: np.ndarray = np.linspace(-0.1, 0.1)
eta_s_2: np.ndarray = np.linspace(-0.1, 0.035)
i_1: np.ndarray = butler_volmer(i_0=10, alpha_a=0.5, alpha_c=0.5, temp=293.15, eta_s=eta_s_1)
i_3: np.ndarray = butler_volmer(i_0=5, alpha_a=0.5, alpha_c=0.5, temp=293.15, eta_s=eta_s_1)
i_2: np.ndarray = butler_volmer(i_0=10, alpha_a=1.5, alpha_c=0.5, temp=293.15, eta_s=eta_s_2)

plt.plot(eta_s_1, i_1, label="eta_1")
plt.plot(eta_s_2, i_2, label="eta_2")
plt.plot(eta_s_1, i_3, label="eta_3")

plt.legend()
plt.show()

