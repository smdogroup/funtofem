

__all__ = ["Analysis", "Aeroelastic", "Aerothermal", "Aerothermoelastic"]

from typing import TYPE_CHECKING

class Analysis:
    def __init__(self, name:str, dynamic:bool=False, use_elastic:bool=True, use_thermal:bool=True):
        self._name = name
        self._dynamic = dynamic

        # need at least one to be true to have a coupled analysis
        assert(self._use_elastic or self._use_thermal)
        self._use_elastic = use_elastic
        self._use_thermal = use_thermal

    @property
    def dynamic_str(self) -> str:
        if self._dynamic:
            return "Dynamic"
        else:
            return "Steady"

    @property
    def analysis_type(self) -> str:
        if self._use_elastic and self._use_thermal:
            return f"{self.dynamic_str}-Aerothermoelastic"
        elif self._use_elastic and not self._use_thermal:
            return f"{self.dynamic_str}-Aeroelastic"
        elif self._use_thermal and not self._use_elastic:
            return f"{self.dynamic_str}-Aerothermal"
        else:
            raise AssertionError("Unsupported analysis type must have elastic or thermal to proceed.")

class Aeroelastic(Analysis):
    def __init__(self, name:str, dynamic:bool=False):
        super(Aeroelastic,self).__init__(name=name, dynamic=dynamic, use_elastic=True, use_thermal=False)

class Aerothermal(Analysis):
    def __init__(self, name:str, dynamic:bool=False):
        super(Aerothermal,self).__init__(name=name, dynamic=dynamic, use_elastic=False, use_thermal=True)

class Aerothermoelastic(Analysis):
    def __init__(self, name:str, dynamic:bool=False):
        super(Aerothermoelastic,self).__init__(name=name, dynamic=dynamic, use_elastic=True, use_thermal=True)