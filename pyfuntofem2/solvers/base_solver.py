
__all__ = ["Solver", "StructureSolver", "FluidSolver"]

from typing import TYPE_CHECKING

class Solver:
    def __init__(self, name):
        self._name = name

class StructureSolver(Solver):
    def __init__(self, name):
        self._name = name

class FluidSolver(Solver):
    def __init__(self, name):
        self._name = name