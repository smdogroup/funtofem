
__all__ = ["SolverManager", "CommManager"]

from typing import TYPE_CHECKING
from .tacs_interface import TacsSteadyInterface
from .tacs_interface_unsteady import TacsUnsteadyInterface

class CommManager:
    def __init__(self, master_comm, struct_comm=None, struct_root=0, aero_comm=None, aero_root=0):
        """
        Comm Manager holds the disciplinary comms of each solver below in the SolverManager class

        Parameters
        --------------------------------------------------
        master_comm : MPI.COMM, regular communicator
        struct_comm : MPI.COMM, partitioned struct communicator
        struct_root : int, root proc ind for structure solver
        aero_comm : MPI.COMM, partitioned aero communicator
        aero_root : int, root proc ind for aero solver
        """
        self.master_comm = master_comm
        if struct_comm is not None:
            self.struct_comm = struct_comm
        else:
            self.struct_comm = master_comm
        self.struct_root = struct_root
        if aero_comm is not None:
            self.aero_comm = aero_comm
        else:
            self.aero_comm = master_comm
        self.aero_root = aero_root

class SolverManager:
    def __init__(self, comm, use_flow:bool=True, use_struct:bool=True):
        """
        Create a solver manager object which holds flow, struct solvers
        and in the future might be expanded to hold dynamics, etc.

        Parameters
        ---------------------------------------------------
        comm: MPI COMM
            MPI master communicator
        use_flow: bool
            whether to require flow solvers like Fun3dInterface
        use_struct: bool
            whether to require structural solvers like TacsInterface
        """
        self.comm = comm
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
    def comm_manager(self) -> CommManager:
        """
        make a default comm manager from the discipline solvers
        """
        return CommManager(
            master_comm=self.comm,
            struct_comm=self.struct_comm,
            struct_root=self.struct_root,
            aero_comm=self.aero_comm,
            aero_root=self.aero_root,
        )

    @property
    def aero_comm(self):
        return self.flow.comm

    @property
    def aero_root(self):
        return 0

    @property
    def struct_comm(self):
        is_tacs = isinstance(self.structural, TacsSteadyInterface) or \
            isinstance(self.structural, TacsUnsteadyInterface)
        if is_tacs: # TODO : change tacs_comm -> comm and comm -> master_comm so simpler
            return self.structural.tacs_comm
        else:
            return self.structural.comm

    @property
    def struct_root(self):
        return 0

    @property
    def fully_defined(self) -> bool:
        has_flow = not(self.use_flow) or self.flow is not None
        has_struct = not(self.use_struct) or self.structural is not None
        return has_flow and has_struct
