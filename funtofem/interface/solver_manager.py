from __future__ import annotations

__all__ = ["SolverManager", "CommManager"]

from typing import TYPE_CHECKING
import importlib.util

tacs_loader = importlib.util.find_spec("tacs")
if tacs_loader is not None:
    from .tacs_interface import TacsSteadyInterface
    from .tacs_interface_unsteady import TacsUnsteadyInterface


fun3d_loader = importlib.util.find_spec("fun3d")
if fun3d_loader is not None:
    from .fun3d_14_interface import Fun3d14Interface

if TYPE_CHECKING:
    from mpi4py import MPI


class CommManager:
    def __init__(
        self,
        master_comm: MPI.Comm,
        struct_comm: MPI.Comm = None,
        struct_root=0,
        aero_comm: MPI.Comm = None,
        aero_root=0,
    ):
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

    def __str__(self):
        line0 = f"CommManager"
        line1 = f"  Master comm: {self.master_comm}"
        line2 = f"  Aero comm: {self.aero_comm}"
        line3 = f"    Aero root: {self.aero_root}"
        line4 = f"  Struct comm: {self.struct_comm}"
        line5 = f"    Struct root: {self.struct_root}"

        output = (line0, line1, line2, line3, line4, line5)

        return "\n".join(output)


class SolverManager:
    def __init__(self, comm: MPI.Comm, use_flow: bool = True, use_struct: bool = True):
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
    def uses_fun3d(self) -> bool:
        if fun3d_loader is None or self.flow is None:
            return False
        else:
            return isinstance(self.flow, Fun3d14Interface)

    @property
    def solver_list(self):
        """
        return a list of solvers
        """
        mlist: list[Fun3d14Interface | TacsSteadyInterface] = []
        if self.use_flow:
            mlist.append(self.flow)
        if self.use_struct:
            mlist.append(self.structural)
        return mlist

    @property
    def forward_residual(self) -> float:
        return max([abs(solver.get_forward_residual()) for solver in self.solver_list])

    @property
    def adjoint_residual(self) -> float:
        return max([abs(solver.get_adjoint_residual()) for solver in self.solver_list])

    @property
    def flow(self) -> Fun3d14Interface:
        return self._flow

    @flow.setter
    def flow(self, new_flow_solver):
        self._flow = new_flow_solver

    def make_flow_real(self):
        """
        switch fun3d flow to real
        """
        self.flow = Fun3d14Interface.copy_real_interface(self.flow)
        return self

    def make_flow_complex(self):
        """
        switch fun3d flow to complex
        """
        print(f"inside make flow complex", flush=True)
        self.flow = Fun3d14Interface.copy_complex_interface(self.flow)
        return self

    @property
    def structural(self) -> TacsSteadyInterface | TacsUnsteadyInterface:
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
        if self.flow is not None:
            return self.flow.comm
        else:
            return self.comm

    @property
    def aero_root(self):
        return 0

    @property
    def struct_comm(self):
        if tacs_loader is not None:
            if isinstance(self.structural, TacsSteadyInterface) or isinstance(
                self.structural, TacsUnsteadyInterface
            ):
                return self.structural.tacs_comm
        elif self.structural is not None:
            return self.structural.comm
        else:
            return self.comm

    @property
    def struct_root(self):
        return 0

    @property
    def fully_defined(self) -> bool:
        has_flow = not (self.use_flow) or self.flow is not None
        has_struct = not (self.use_struct) or self.structural is not None
        return has_flow and has_struct

    def print_summary(self, comm=None, filename=None):
        """
        Print a summary of the SolverManager and each registered solver.

        Parameters
        ----------
        comm : MPI communicator, optional
            If provided, only rank 0 prints and barriers are inserted.
            Defaults to self.comm.
        filename : str or path-like, optional
            Write the full summary (manager + all solvers) to this file
            (opened in write mode) instead of stdout.
        """
        comm = comm if comm is not None else self.comm

        print_here = True
        if comm is not None:
            comm.Barrier()
            if comm.rank != 0:
                print_here = False

        if print_here:
            if filename is not None:
                fp = open(filename, "w")
            else:
                fp = None

            p = lambda *args, **kw: print(*args, file=fp, **kw)

            p("==========================================================")
            p("||            Solver Manager Summary                   ||")
            p("==========================================================")

            flow_type = type(self.flow).__name__ if self.flow is not None else "None"
            struct_type = (
                type(self.structural).__name__
                if self.structural is not None
                else "None"
            )

            p(f"  Flow solver          : {flow_type}")
            p(f"  Structural solver    : {struct_type}")
            p(f"  Uses FUN3D           : {self.uses_fun3d}")

            # --- Delegate to each solver, sharing the open file handle ---
            if self.flow is not None and hasattr(self.flow, "print_summary"):
                p("")
                self.flow.print_summary(_fp=fp)

            if self.structural is not None and hasattr(
                self.structural, "print_summary"
            ):
                p("")
                self.structural.print_summary(_fp=fp)

            if fp is not None:
                fp.close()

        if comm is not None:
            comm.Barrier()

        return
