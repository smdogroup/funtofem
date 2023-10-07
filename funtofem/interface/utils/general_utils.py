# -------------------------------------------------
# General utility functions used in FUNtoFEM.
# The import of these should not be based on the availability of any loader.
# -------------------------------------------------

__all__ = [
    "real_norm",
    "imag_norm",
]

import numpy as np


def real_norm(vec):
    if vec is None:
        return None
    return np.linalg.norm(np.real(vec))


def imag_norm(vec):
    if vec is None:
        return None
    return np.linalg.norm(np.imag(vec))
