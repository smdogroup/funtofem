

__all__ = ["CoupledModel"]

from pyfuntofem2.geometry.body import Body
from pyfuntofem2.geometry.fluid_volume import FluidVolume
from ..problem.functions import StructFunction
from pyfuntofem2.solvers.base_solver import FluidSolver, StructureSolver

class CoupledModel:
    def __init__(self, body:Body, fluid_volume:FluidVolume):
        self._body = body
        self._fluid_volume = fluid_volume
        self._structure_solver = None
        self._fluid_solver = None

    @property
    def fluid_solver(self) -> FluidSolver:
        return self._fluid_solver

    @fluid_solver.setter
    def fluid_solver(self, solver:FluidSolver):
        self._fluid_solver = solver
        
    @property
    def structure_solver(self) -> StructureSolver:
        return self._structure_solver

    @structure_solver.setter
    def structure_solver(self, solver:StructureSolver):
        self._structure_solver = solver