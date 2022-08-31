

__all__ = ["Body"]

from typing import TYPE_CHECKING, List
import numpy as np
from pyfuntofem2.problem.variable import Variable, StructVariable
from pyfuntofem2.solvers.tacs_solver import TACSInterface

class Body:
    """
    Body class which represents the structural geometry
    TODO : add tacs interface components here or a separate class in this file
    """

    MOTION_TYPES = ["deform", "rigid+deform", "rigid", "deform+rigid"]
    def __init__(self, name:str, boundary_idx:int=0, use_fun3d:bool=True, motion_type:str="deform"):
        self._name = name
        self._boundary_idx = boundary_idx
        self._use_fun3d = use_fun3d

        self.motion_type = motion_type

        self._xfer_ndof = 3
        self._therm_ndof = 1
        self._thermal_idx = 3
        self._T_ref = 300.0 # Kelvin

        self._variables = []

        self._solver = None

    @property
    def variables(self) -> List[StructVariable]:
        return self._variables

    @variables.setter
    def variables(self, new_variables):
        for new_var in new_variables:
            # only allow struct variables to be stored in the body
            assert(isinstance(new_var, StructVariable))
        self._variables = new_variables

    def add_variables(self, variables:StructVariable or List[Variable]):
        if isinstance(variables, Variable):
            single_var = variables
            self.variables += [single_var]
        elif isinstance(variables, list):
            self.variables += variables
        else:
            raise AssertionError("Attempted to add variables that are not Variable objects!")

    @property
    def name(self) -> str:
        return self._name

    @property
    def use_fun3d(self) -> bool:
        return self._use_fun3d

    @property
    def motion_type(self) -> str:
        return self._motion_type

    @motion_type.setter
    def motion_type(self, the_type:str):
        assert(the_type in Body.MOTION_TYPES)
        self._motion_type = the_type

    @property
    def soolver()


       

    

    