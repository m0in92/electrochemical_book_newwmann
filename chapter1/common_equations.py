from typing import Union

import numpy as np
import numpy.typing as npt

from utils import PhysicsConstants


def BV(i_0: float, alpha_a: float, alpha_c: float, temp: float, eta_s: Union[float, npt.ArrayLike]) -> Union[float, npt.ArrayLike]:
    """
    Calculates the current density [A/m2] using the Butler-Volmer equation
    :param i_0: exchange current density [A/m2]
    :param alpha_a: apparent transfer coefficient for the anodic reaction
    :param alpha_c: apparent transfer coefficient for the cathodic reaction
    :param temp: temperature [K]
    :param eta_s: electrode surface over-potential [V]
    :return:
    """
    return i_0 * (np.exp(alpha_a * PhysicsConstants.F * eta_s/(PhysicsConstants.R * temp)) - \
                  np.exp(-alpha_c * PhysicsConstants.F * eta_s/(PhysicsConstants.R * temp)))
