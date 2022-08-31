
#!/usr/bin/env python

# This file is part of the package FUNtoFEM for coupled aeroelastic simulation
# and design optimization.

# Copyright (C) 2015 Georgia Tech Research Corporation.
# Additional copyright (C) 2015 Kevin Jacobson, Jan Kiviaho and Graeme Kennedy.
# All rights reserved.

# FUNtoFEM is licensed under the Apache License, Version 2.0 (the "License");
# you may not use this software except in compliance with the License.
# You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = ["Variable", "AeroVariable", "StructVariable"]

class Variable:
    def __init__(self, name:str, value:float=0.0, lb:float=0.0, ub:float=1.0, scaling:float=1.0, active:bool=True, coupled:bool=False, id:int=0):
        self._name = name
        self._value = value
        self._lb = lb
        self._ub = ub
        self._scaling = scaling
        self._coupled = coupled
        self._id = id
    
    @property
    def name(self) -> str:
        return self._name

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, new_value:float):
        self._value = new_value

    def __str__(self):
        return f"'{self.name}' variable with value={self.value}"

class AeroVariable(Variable):
    """
    Aerodynamic Design Variable
    """
    def __init__(self, name:str, value:float=0.0, lb:float=0.0, ub:float=1.0, scaling:float=1.0, active:bool=True, coupled:bool=False, id:int=0):
        super(AeroVariable,self).__init__(name=name,value=value, lb=lb, ub=ub, scaling=scaling, active=active,coupled=coupled, id=id)

    def __str__(self):
        return f"'{self.name}' aerodynamic variable with value={self.value}"

class StructVariable(Variable):
    """
    Structure Design Variable
    """
    def __init__(self, name:str, value:float=0.0, lb:float=0.0, ub:float=1.0, scaling:float=1.0, active:bool=True, coupled:bool=False, id:int=0):
        super(StructVariable,self).__init__(name=name,value=value, lb=lb, ub=ub, scaling=scaling, active=active,coupled=coupled, id=id)

    def __str__(self):
        return f"'{self.name}' structural variable with value={self.value}"