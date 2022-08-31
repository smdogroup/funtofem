

from pyfuntofem2.solvers.base_solver import FluidSolver
from pyfuntofem2.geometry.fluid_volume import FluidVolume

from typing import TYPE_CHECKING

class Fun3dSolver(FluidSolver):
    def __init__(self, name:str, fluid_volume:FluidVolume, dat_file:str):
        self._name =  name
        self._fluid_volume = fluid_volume
        # TODO : make custom path object here
        self._dat_file = dat_file