
__all__ = ["SolverManager"]

from typing import TYPE_CHECKING

class SolverManager:
    def __init__(self, use_flow:bool=True, use_struct:bool=True):
        """
        Create a solver manager object which holds flow, struct solvers
        and in the future might be expanded to hold dynamics, etc.

        Parameters
        ---------------------------------------------------
        use_flow: bool
            whether to require flow solvers like Fun3dInterface
        use_struct: bool
            whether to require structural solvers like TacsInterface
        """
        self._use_flow = use_flow
        self._use_struct = use_struct

        self._flow = None
        self._structural = None

    @property
    def use_flow(self) -> bool:
        return self._flow

    @property
    def use_struct(self) -> bool:
        return self._use_struct

    @property
    def solver_list(self):
        """
        return a list of solvers
        """
        mlist = []
        if self.use_flow:
            mlist.append(self.flow)
        if self.use_struct:
            mlist.append(self.structural)
        return mlist

    @property
    def flow(self):
        return self._flow

    @flow.setter
    def flow(self, new_flow_solver):
        self._flow = new_flow_solver

    @property
    def structural(self):
        return self._structural

    @structural.setter
    def structural(self, new_structural_solver):
        self._structural = new_structural_solver

    @property
    def fully_defined(self) -> bool:
        has_flow = not(self.use_flow) or self.flow is not None
        has_struct = not(self.use_struct) or self.structural is not None
        return has_flow and has_struct
