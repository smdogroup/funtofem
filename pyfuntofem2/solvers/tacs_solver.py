

__all__ = ["TacsSolver"]

from pyfuntofem2.solvers.base_solver import StructureSolver
from pyfuntofem2.geometry.body import Body

class TacsSolver(StructureSolver):
    def __init__(self, body:Body, dat_file:str):
        self._body = body
        self._dat_file = dat_file