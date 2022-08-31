
__all__ = ["FluidVolume"]

from ast import Str
from typing import TYPE_CHECKING, List
import numpy as np
from pyfuntofem2.problem.variable import Variable, AeroVariable


class FluidVolume:
    """
    FluidVolume represents a fluid geometry and handles aeroDVs
    TODO : add fun3d interface to this portion and/or another class here
    """
    def __init__(self, name:str):
        self._name = name
        self._variables = []

    @property
    def variables(self) -> List[AeroVariable]:
        return self._variables

    @variables.setter
    def variables(self, new_variables):
        for new_var in new_variables:
            # only allow struct variables to be stored in the body
            assert(isinstance(new_var, AeroVariable))
        self._variables = new_variables

    def add_variables(self, vars:AeroVariable or List[AeroVariable]):
        if isinstance(vars, Variable):
            single_var = vars
            self.variables += [single_var]
        elif isinstance(vars, list):
            self.variables += vars
        else:
            raise AssertionError("Attempted to add variables that are not Variable objects!")

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new_name:str):
        self._name = new_name
       

    

    