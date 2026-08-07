#!/usr/bin/env python
"""
This file is part of the package FUNtoFEM for coupled aeroelastic simulation
and design optimization.

Copyright (C) 2015 Georgia Tech Research Corporation.
Additional copyright (C) 2015 Kevin Jacobson, Jan Kiviaho and Graeme Kennedy.
All rights reserved.

FUNtoFEM is licensed under the Apache License, Version 2.0 (the "License");
you may not use this software except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

__all__ = ["Scenario"]

from ._base import Base
from .variable import Variable
from .function import Function
import numpy as np
import importlib
from typing import TYPE_CHECKING

tacs_loader = importlib.util.find_spec("tacs")
if tacs_loader is not None:
    from funtofem.interface import TacsIntegrationSettings

if TYPE_CHECKING:
    from .composite_function import CompositeFunction


def _on_root_proc() -> bool:
    """whether this is the root proc, so status messages only print once under MPI"""
    try:
        from mpi4py import MPI

        return MPI.COMM_WORLD.rank == 0
    except ImportError:  # pragma: no cover - funtofem normally requires mpi4py
        return True


class Scenario(Base):
    """A class to hold scenario information for a design point in optimization"""

    UNCOUPLED_STEP_BUFFER = 10

    # Automatically reorder function list on add_function so that all functions that require
    # an adjoint come first (also, when early stopping is active, an aerodynamic function
    # comes first). False: hard error when functions are registered out of order.
    # See _canonicalize_functions.
    AUTO_REORDER_FUNCTIONS = True

    def __init__(
        self,
        name: str,
        id=0,
        group=None,
        steady=True,
        fun3d=True,
        steps=1000,
        uncoupled_steps=0,
        coupled=True,
        adjoint_steps=None,
        min_forward_steps=50,
        min_adjoint_steps=None,
        forward_coupling_frequency=1,
        adjoint_coupling_frequency=1,
        early_stopping=False,
        post_tight_forward_steps=0,
        post_tight_adjoint_steps=0,
        post_forward_coupling_freq=1,
        post_adjoint_coupling_freq=1,
        T_ref=300,
        T_inf=300,
        qinf=1.0,
        flow_dt=1.0,
        tacs_integration_settings=None,
        fun3d_project_name=None,
        suther1=1.458205e-6,
        suther2=110.3333,
        gamma=1.4,
        R_specific=287.058,
        Pr=0.72,
        Mach_inf=None,
        turbulent=True,
        k_fixed=None,
        T_fixed=None,
    ):
        """
        Parameters
        ----------
        name: str
            name of the scenario
        id: int
            ID number of the body in the list of bodies in the model
        group: int
            group number for the scenario. Coupled variables defined in the scenario will be coupled with
            scenarios in the same group
        steady: bool
            whether the scenario's simulation is steady or unsteady
        fun3d: bool
            whether or not you are using FUN3D. If true, the scenario class will auto-populate 'aerodynamic' required by FUN3D
        steps: int
            the number of outer coupling steps in the scenario
        uncoupled_steps: int
            the number of fun3d iterations ran before coupled iterations
        coupled: bool
            Whether this scenario is a fully coupled aerostructural scenario (default True). Set to False
            for uncoupled scenarios, e.g., when using OnewayStructDriver with pre-computed aerodynamic
            loads. Uncoupled scenarios skip the fluid-structure coupling loop and are used by OnewayStructTrimDriver
            to identify which scenarios to apply AOA load scaling to.
        adjoint_steps: int
            optional number of adjoint steps when using FUN3D analysis, can have different
            number of forward and adjoint steps in steady-state
        forward_coupling_frequency: int
            the number of uncoupled flow iterations per coupled iteration in the forward analysis
            e.g. with FUN3D the total max number of FUN3D steps is steps * forward_coupling_frequency + uncoupled_steps
        adjoint_coupling_frequency: int
            the number of uncoupled flow adjoint iterations per coupled iteration in the adjoint analysis
            e.g. with FUN3D the total max number of FUN3D adjoint steps is adjoint_steps * adjoint_coupling_frequency
        early_stopping: bool
            whether to activate the early stopping criterion
        min_forward_steps: int
            (optional) minimum number of steps required before early stopping can happen. Note
            this is set to the # of uncoupled steps if not provided (hence you probably don't need to set this
            but you can in special circumstances)
        min_adjoint_steps: int
            (optional) minimum number of adjoint steps required before early stopping criterion is applied
        post_tight_forward_steps: int
            (optional) number of additional tightly coupled forward steps at the end of the solve
        post_tight_adjoint_steps: int
            (optional) number of additional tightly coupled adjoint steps at the end of the solve
        T_ref: double
            Structural reference temperature (i.e., unperturbed temperature of structure) in Kelvin.
        T_inf: double
            Fluid freestream reference temperature in Kelvin.
        qinf: float
            elastic load dimensionalization factor = 0.5 * rho_inf * v_inf^2
        flow_dt: float
            Equals the nondimensional time step in fun3d.nml (time_step_nondim)
        tacs_integration_settings: :class:`~interface.TacsUnsteadyInterface`
            Optional TacsIntegrator settings for the unsteady interface (required for unsteady)
        fun3d_project_name: filename
            name of project_rootname from fun3d.nml, ex: funtofem_CAPS would have a grid file funtofem_CAPS.lb8.ugrid

        Optional Parameters/Constants
        -----------------------------
        suther1: double
            First constant in Sutherland's two-constant viscosity model. Units of kg/m-s-K^0.5
        suther2: double
            Second constant in Sutherland's two-constant viscosity model. Units of K
        gamma: double
            Ratio of specific heats.
        R_specific: double
            Specific gas constant of the working fluid (assumed air). Units of J/kg-K
        Pr: double
            Prandtl number.
        Mach_inf: float or None
            Freestream Mach number. When provided, selects the ``"eckert"`` k-evaluation
            strategy: thermal conductivity is evaluated at the Eckert reference temperature
            T* = 0.5*(T_w + T_inf) + 0.22*(T_aw - T_inf), which severs the positive-feedback
            loop that destabilizes the aerothermal coupling at high Mach numbers. When None,
            falls back to the legacy ``"wall"`` strategy (k evaluated at T_wall) with a
            one-time warning. Ignored when ``k_fixed`` is also supplied.
        turbulent: bool
            Recovery factor formulation used in the Eckert adiabatic-wall temperature.
            True (default) uses the turbulent form r = Pr^(1/3); False uses the laminar
            form r = sqrt(Pr). Only relevant when ``Mach_inf`` is set.
        k_fixed: float or None
            When provided, selects the ``"fixed"`` k-evaluation strategy: thermal conductivity
            is held at this constant value (W/m-K) for every coupling exchange. This is the
            only provably globally-contractive strategy and is useful for debugging or as a
            conservative fallback. Takes precedence over ``Mach_inf`` and ``T_fixed``.
        T_fixed: float or None
            Reference temperature (K) at which Sutherland's law is evaluated once to produce
            a constant k for the ``"fixed"`` strategy. Convenient when you want to pin k at a
            physically meaningful temperature without computing the value by hand. Ignored when
            ``k_fixed`` is also supplied.
        See Also
        --------
        :mod:`base` : Scenario inherits from Base
        """

        super(Scenario, self).__init__(name, id, group)

        self.name = name
        self.id = id
        self.group = group
        self.group_master = False
        self._adjoint_steps = adjoint_steps
        self.variables = {}

        self.functions = []
        # whether the function list has already been reordered once (so the
        # notice about it is only printed once per scenario)
        self._reordered = False
        self.coupled = coupled
        self.steady = steady
        self.steps = steps
        self.forward_coupling_frequency = forward_coupling_frequency
        self.adjoint_coupling_frequency = adjoint_coupling_frequency
        self.uncoupled_steps = uncoupled_steps
        self.post_tight_forward_steps = post_tight_forward_steps
        self.post_tight_adjoint_steps = post_tight_adjoint_steps
        self.post_forward_coupling_freq = post_forward_coupling_freq
        self.post_adjoint_coupling_freq = post_adjoint_coupling_freq

        self.tacs_integration_settings = tacs_integration_settings
        self.fun3d_project_name = fun3d_project_name

        self.T_ref = T_ref
        self.T_inf = T_inf
        self.qinf = qinf
        self.flow_dt = flow_dt

        self.suther1 = suther1
        self.suther2 = suther2
        self.gamma = gamma
        self.R_specific = R_specific
        self.Pr = Pr

        # Heat capacity at constant pressure — must be set before set_conductivity_info,
        # which may call _sutherland_k (used when T_fixed is supplied).
        self.cp = self.R_specific * self.gamma / (self.gamma - 1)

        self.set_conductivity_info(
            Mach_inf=Mach_inf, turbulent=turbulent, k_fixed=k_fixed, T_fixed=T_fixed
        )

        self.coupled_fw_rtol = 1e-6
        self.coupled_adj_rtol = 1e-6

        # early stopping criterion
        self.min_forward_steps = (
            min_forward_steps
            if min_forward_steps is not None
            else uncoupled_steps + self.UNCOUPLED_STEP_BUFFER
        )
        self.min_adjoint_steps = (
            min_adjoint_steps if min_adjoint_steps is not None else 0
        )
        self.early_stopping = early_stopping

        if fun3d:
            mach = Variable("Mach", id=1, upper=5.0, active=False)
            aoa = Variable("AOA", id=2, lower=-15.0, upper=15.0, active=False)
            yaw = Variable("Yaw", id=3, lower=-10.0, upper=10.0, active=False)
            xrate = Variable("xrate", id=4, upper=0.0, active=False)
            yrate = Variable("yrate", id=5, upper=0.0, active=False)
            zrate = Variable("zrate", id=6, upper=0.0, active=False)

            self.add_variable("aerodynamic", mach)
            self.add_variable("aerodynamic", aoa)
            self.add_variable("aerodynamic", yaw)
            self.add_variable("aerodynamic", xrate)
            self.add_variable("aerodynamic", yrate)
            self.add_variable("aerodynamic", zrate)

    @classmethod
    def steady(
        cls,
        name: str,
        steps: int,
        coupled: bool = True,
        uncoupled_steps: int = 0,
        forward_coupling_frequency: int = 1,
        adjoint_coupling_frequency: int = 1,
        adjoint_steps: int = None,
        post_tight_forward_steps=0,
        post_tight_adjoint_steps=0,
    ):
        return cls(
            name=name,
            steady=True,
            steps=steps,
            coupled=coupled,
            forward_coupling_frequency=forward_coupling_frequency,
            adjoint_steps=adjoint_steps,
            adjoint_coupling_frequency=adjoint_coupling_frequency,
            uncoupled_steps=uncoupled_steps,
            post_tight_forward_steps=post_tight_forward_steps,
            post_tight_adjoint_steps=post_tight_adjoint_steps,
        )

    @classmethod
    def unsteady(
        cls,
        name: str,
        steps: int,
        coupled: bool = True,
        uncoupled_steps: int = 0,
        tacs_integration_settings=None,
    ):
        return cls(
            name=name,
            steady=False,
            steps=steps,
            coupled=coupled,
            tacs_integration_settings=tacs_integration_settings,
            uncoupled_steps=uncoupled_steps,
        )

    @property
    def adjoint_steps(self) -> int:
        """
        in the steady case it's best to choose the
        adjoint steps based on the funtofem coupling frequency
        """
        if self._adjoint_steps is not None and self.steady:
            return self._adjoint_steps
        elif not self.steady:
            return None  # defaults to number of steps in unsteady case
        else:  # choose it based on funtofem coupling frequency in steady case
            return int(np.ceil(self.steps / self.adjoint_coupling_frequency))

    @adjoint_steps.setter
    def adjoint_steps(self, new_steps: int):
        assert self.steady
        self._adjoint_steps = new_steps

    @property
    def early_stopping(self) -> bool:
        return self._early_stopping

    @early_stopping.setter
    def early_stopping(self, value: bool):
        self._early_stopping = value
        # the required function ordering depends on this setting, and it may be
        # changed after functions have already been registered
        self._canonicalize_functions()

    def _canonicalize_functions(self):
        """
        Reorder `self.functions` in place into the canonical order the rest of the
        framework assumes, and renumber the function ids to match.

        Two orderings are required downstream:

        1. All functions requiring an adjoint must come first. The adjoint-Jacobian
           product arrays in Body are sized with `count_adjoint_functions()` while much
           of the driver and interface code indexes them with the full function index,
           so the two index spaces must coincide. Critically, `function.id` is the
           1-based full-list index and is pushed straight into FUN3D's Fortran design
           interface, which was sized with `count_adjoint_functions()` -- getting this
           wrong is an out-of-bounds write into compiled code, not a Python error.

        2. When the early stopping criterion is on, an aerodynamic function must come
           first, otherwise FUN3D's adjoint early stopping criterion fails (see
           `Fun3d14Interface.set_functions`).

        The sort is stable, so the order in which the user registered functions is
        preserved within each group. `add_function` keeps the list canonical as it goes
        and only calls this when an append would actually break the invariant.
        """
        old_order = list(self.functions)

        if self.AUTO_REORDER_FUNCTIONS:
            # adjoint functions first
            self.functions.sort(key=lambda func: not func.adjoint)

            # then, if early stopping is on and there is an aerodynamic function to
            # promote, move an aerodynamic one to the front of the adjoint group
            if self.early_stopping and any(
                func.analysis_type == "aerodynamic" for func in self.adjoint_functions
            ):
                self.functions.sort(
                    key=lambda func: (
                        not func.adjoint,
                        func.analysis_type != "aerodynamic",
                    )
                )

        # renumber so function.id stays the 1-based index into self.functions
        for ifunc, func in enumerate(self.functions):
            func.id = ifunc + 1

        if self.functions != old_order and not self._reordered:
            self._reordered = True
            if _on_root_proc():
                print(
                    f"FUNtoFEM scenario '{self.name}': functions were registered out of "
                    "order and are being reordered automatically so that functions "
                    "requiring an adjoint come first. scenario.functions is therefore "
                    "not in registration order; call scenario.print_summary() to see "
                    "the order that will be used.",
                    flush=True,
                )
        return

    def add_function(self, function: Function):
        """
        Add a new function to the scenario's function list

        Functions may be registered in any order; the list is reordered internally so
        that all functions requiring an adjoint come first (see
        `_canonicalize_functions`). Set `Scenario.AUTO_REORDER_FUNCTIONS = False` to
        get a hard error on out-of-order registration instead.

        Parameters
        ----------
        function: Function
            function object to be added to scenario
        """

        if not isinstance(function, Function):
            raise TypeError(
                f"FUNtoFEM scenario '{self.name}': add_function expects a Function, got "
                f"{type(function).__name__}. Composite functions are registered to the "
                "model rather than to a scenario, via "
                "CompositeFunction.register_to(model)."
            )

        if not self.AUTO_REORDER_FUNCTIONS and function.adjoint:
            blockers = [
                (ifunc, func)
                for ifunc, func in enumerate(self.functions)
                if not func.adjoint
            ]
            if blockers:
                ifunc, blocker = blockers[0]
                listing = "\n".join(
                    f"    [{jfunc}] {func.name:<16} adjoint={func.adjoint}"
                    for jfunc, func in enumerate(self.functions)
                )
                raise RuntimeError(
                    f"FUNtoFEM: cannot register adjoint function '{function.name}' to "
                    f"scenario '{self.name}' after non-adjoint function "
                    f"'{blocker.name}' (index {ifunc}). All functions requiring an "
                    "adjoint must be registered first.\n\n"
                    f"  Current order in scenario '{self.name}':\n"
                    f"{listing}\n"
                    f"    [{len(self.functions)}] {function.name:<16} "
                    f"adjoint={function.adjoint}   <-- rejected\n\n"
                    f"  Fix: register '{blocker.name}' last, or leave "
                    "Scenario.AUTO_REORDER_FUNCTIONS = True (the default) to have "
                    "FUNtoFEM reorder them automatically."
                )

        function.scenario = self.id
        function._scenario_name = self.name

        # self.functions is canonical before every add (this method maintains that
        # invariant): a non-adjoint function always belongs at the end, and an
        # adjoint function belongs at the end only if nothing non-adjoint is there
        # yet -- which is true exactly when the current last function is adjoint.
        # Early stopping additionally wants an aerodynamic function promoted to the
        # front, which is not a local decision, so fall through in that case.
        previous_last = self.functions[-1] if self.functions else None
        self.functions.append(function)

        stays_canonical = not self.early_stopping and (
            not function.adjoint or previous_last is None or previous_last.adjoint
        )
        if stays_canonical:
            function.id = len(self.functions)
        else:
            self._canonicalize_functions()

        # return the object for method cascading
        return self

    @property
    def adjoint_functions(self) -> list[Function | CompositeFunction]:
        """return a list of the adjoint functions only"""
        return [func for func in self.functions if func.adjoint]

    @property
    def adjoint_map(self) -> dict:
        """return an int map from adjoint function index to full function list index"""
        adj_dict = {}
        adj_ct = 0
        for ifunc, func in enumerate(self.functions):
            if func.adjoint:
                adj_dict[adj_ct] = ifunc
                adj_ct += 1
        return adj_dict

    @property
    def reverse_adjoint_map(self) -> dict:
        """
        return an int map from full function index to adjoint function index

        Only adjoint functions appear as keys, so callers iterating the full function
        list should skip indices that are absent.
        """
        return {ifunc: iadj for iadj, ifunc in self.adjoint_map.items()}

    def count_functions(self):
        """
        Returns the number of functions in this scenario

        Returns
        -------
        count: int
            number of functions in this scenario

        """
        return len(self.functions)

    def count_adjoint_functions(self):
        """
        Returns the number of functions that require an adjoint in this scenario

        Returns
        -------
        count: int
            number of adjoint-requiring functions in this scenario

        """
        is_adjoint = lambda func: func.adjoint
        return len(list(filter(is_adjoint, self.functions)))

    def add_variable(self, vartype, var: Variable):
        """
        Add a new variable to the scenario's variable dictionary

        Parameters
        ----------
        vartype: str
            type of variable
        var: Variable object
            variable to be added
        """
        # var.scenario = self.id
        var._scenario_name = self.name

        super(Scenario, self).add_variable(vartype, var)

        # return object for method cascading
        return self

    def include(self, obj):
        """
        generic include method adds objects for readability
        """
        if isinstance(obj, Function):
            self.add_function(obj)
        elif isinstance(obj, Variable):
            assert obj.analysis_type is not None
            self.add_variable(vartype=obj.analysis_type, var=obj)
        elif isinstance(obj, TacsIntegrationSettings):
            self.tacs_integration_settings = obj
        else:
            raise ValueError(
                "Scenario include method does not currently support other methods"
            )

        # return the object for method cascading
        return self

    def fun3d_project(self, new_proj_name):
        """set the fun3d project rootname from fun3d.nml for use in shape drivers"""
        self.fun3d_project_name = new_proj_name
        return self

    def register_to(self, funtofem_model):
        """
        add this scenario to the funtofem model at the end of a method cascade
        """
        funtofem_model.add_scenario(self)
        return self

    def set_temperature(self, T_ref: float = 300.0, T_inf: float = 300.0):
        """
        set structure temperature T_ref and freestream T_inf
        """
        self.T_ref = T_ref
        self.T_inf = T_inf
        return self

    def set_conductivity_info(
        self,
        Mach_inf: float = None,
        turbulent: bool = True,
        k_fixed: float = None,
        T_fixed: float = None,
    ):
        """
        Set the thermal-conductivity evaluation strategy for aerothermal coupling.

        This method provides a method-cascade-friendly alternative to supplying
        ``Mach_inf``, ``turbulent``, and ``k_fixed`` directly to ``__init__``.
        Calling it overwrites any strategy previously set on this scenario.

        The active strategy is determined by which arguments are provided:

        * ``k_fixed`` supplied → ``"fixed"`` strategy: conductivity is held at the
          given constant value (W/m-K) for every coupling exchange.  This is the
          only provably globally-contractive option and is useful for debugging or
          as a conservative fallback, but introduces a steady-state bias.
          Takes precedence over ``Mach_inf`` if both are given.

        * ``T_fixed`` supplied (and ``k_fixed`` is None) → ``"fixed"`` strategy:
          conductivity is computed once from Sutherland's law at the given temperature
          and then held constant.  Convenient when you want to pin k at a physically
          meaningful reference temperature (e.g. the freestream or film temperature)
          without computing the value by hand.  ``k_fixed`` takes precedence if both
          are given.

        * ``Mach_inf`` supplied (and neither ``k_fixed`` nor ``T_fixed``) → ``"eckert"`` strategy:
          conductivity is evaluated at the Eckert reference temperature

              T* = 0.5*(T_w + T_inf) + 0.22*(T_aw - T_inf)

          where T_aw = T_inf * (1 + r*(𝛾-1)/2 * Mach_inf²) and r is the recovery
          factor (turbulent: r = Pr^(1/3); laminar: r = Pr^(1/2)).  Evaluating k at
          T* instead of T_w severs the positive-feedback loop responsible for
          aerothermal coupling instability at significant Mach numbers.

        * Neither supplied → ``"wall"`` strategy (legacy): conductivity is evaluated
          at the current wall temperature T_w.  This is the pre-existing default
          behaviour and is known to be unstable in aerothermal coupling at high Mach
          numbers or low coupling frequency.  A one-time ``UserWarning`` is emitted
          on the first call to ``get_thermal_conduct``.

        Parameters
        ----------
        Mach_inf : float or None
            Freestream Mach number.  Required for the ``"eckert"`` strategy.
        turbulent : bool
            Recovery factor formulation used in the Eckert adiabatic-wall temperature.
            ``True`` (default) uses the turbulent form r = Pr^(1/3); ``False`` uses
            the laminar form r = sqrt(Pr).  Only relevant when ``Mach_inf`` is set.
        k_fixed : float or None
            Constant thermal conductivity value (W/m-K) for the ``"fixed"`` strategy.
            Takes precedence over ``T_fixed`` if both are supplied.
        T_fixed : float or None
            Reference temperature (K) at which to evaluate Sutherland's law once to
            produce a constant k for the ``"fixed"`` strategy.  Ignored when
            ``k_fixed`` is also supplied.

        Returns
        -------
        self : Scenario
            Returns the scenario itself to support method cascading.

        Examples
        --------
        Eckert strategy (recommended for hypersonic aerothermal problems)::

            scenario.set_conductivity_info(Mach_inf=6.47)

        Fixed-k strategy via explicit value::

            scenario.set_conductivity_info(k_fixed=0.05)

        Fixed-k strategy via reference temperature (k computed from Sutherland's law)::

            scenario.set_conductivity_info(T_fixed=241.5)  # e.g. freestream temp

        See Also
        --------
        get_thermal_conduct, get_thermal_conduct_deriv
        """
        # Determine the thermal-conductivity evaluation strategy.
        # "fixed"  : k is held at a constant value (globally contractive).
        # "eckert" : k is evaluated at the Eckert reference temperature T* (recommended
        #            for aerothermal problems at significant Mach numbers).
        # "wall"   : k is evaluated at the current wall temperature T_w (legacy default;
        #            unstable at high Mach / low coupling frequency — use with caution).
        if k_fixed is not None:
            # Explicit k value takes precedence over everything.
            self.k_eval_strategy = "fixed"
            self.k_fixed = float(k_fixed)
            self.Mach_inf = None
            self.turbulent = turbulent
        elif T_fixed is not None:
            # Derive k once from Sutherland's law at the given reference temperature.
            self.k_eval_strategy = "fixed"
            self.k_fixed = float(self._sutherland_k(float(T_fixed)))
            self.Mach_inf = None
            self.turbulent = turbulent
        elif Mach_inf is not None:
            self.k_eval_strategy = "eckert"
            self.k_fixed = None
            self.Mach_inf = float(Mach_inf)
            self.turbulent = bool(turbulent)
        else:
            self.k_eval_strategy = "wall"
            self.k_fixed = None
            self.Mach_inf = None
            self.turbulent = turbulent

        return self

    def set_stop_criterion(
        self,
        early_stopping: bool = True,
        coupled_fw_rtol: float = 1e-6,
        coupled_adj_rtol: float = 1e-6,
        min_forward_steps=None,
        min_adjoint_steps=None,
        post_tight_forward_steps=None,
        post_tight_adjoint_steps=None,
        post_forward_coupling_freq=None,
        post_adjoint_coupling_freq=None,
    ):
        """
        turn on the early stopping criterion, note you probably don't need
        to set the min steps (as it defaults to the # of uncoupled steps)
        The stopping tolerances are set in each discipline interface

        Parameters
        ----------
        early_stopping: bool
            whether to perform early stopping criterion
        min_forward_steps: int
            (optional) - the minimum number of steps for engaging the early stop criterion for forward analysis
        min_adjoint_steps: int
            (optional) - the minimum number of steps for engaging the early stopping criterion for adjoint analysis
        post_tight_forward_steps: int
            (optional) number of additional tightly coupled forward steps at the end of the solve
        post_tight_adjoint_steps: int
            (optional) number of additional tightly coupled adjoint steps at the end of the solve
        """
        self.early_stopping = early_stopping
        self.coupled_fw_rtol = coupled_fw_rtol
        self.coupled_adj_rtol = coupled_adj_rtol
        if min_forward_steps is not None:
            self.min_forward_steps = min_forward_steps
        if min_adjoint_steps is not None:
            self.min_adjoint_steps = min_adjoint_steps
        if post_tight_forward_steps is not None:
            self.post_tight_forward_steps = post_tight_forward_steps
        if post_tight_adjoint_steps is not None:
            self.post_tight_adjoint_steps = post_tight_adjoint_steps
        if post_forward_coupling_freq is not None:
            self.post_forward_coupling_freq = post_forward_coupling_freq
        if post_adjoint_coupling_freq is not None:
            self.post_adjoint_coupling_freq = post_adjoint_coupling_freq
        return self

    def set_flow_ref_vals(self, qinf: float = 1.0, flow_dt: float = 1.0):
        """
        Set flow reference values for FUN3D nondimensionalization.
        flow_dt should always be 1.0 for steady scenarios.

        Parameters
        ----------
        flow_dt: float
            Flow solver time step size. Used to scale the adjoint term coming into and out of
            FUN3D since FUN3D currently uses a different adjoint formulation than FUNtoFEM.
            Should be equal to non-dimensional time step in fun3d.nml (equals time_step_nondim)
        qinf: float
            Dynamic pressure of the freestream flow. Used to nondimensionalize force in FUN3D.
        """

        self.qinf = qinf
        self.flow_dt = flow_dt

        if self.steady is True and float(self.flow_dt) != 1.0:
            raise ValueError("For steady cases, flow_dt must be set to 1.")
        return self

    def set_id(self, id):
        """
        **[model call]**
        Update the id number of the scenario

        Parameters
        ----------

        id: int
            id number of the scenario
        """
        self.id = id

        for vartype in self.variables:
            for var in self.variables[vartype]:
                var.scenario = id

        for func in self.functions:
            func.scenario = id

    def _sutherland_k(self, T):
        """
        Evaluate dimensional thermal conductivity via Sutherland's two-constant viscosity
        law at temperature T (K), using constant Prandtl number and cp.

        Parameters
        ----------
        T : float or np.ndarray
            Temperature(s) at which to evaluate conductivity.  Values are
            floored at 1 K before evaluation so that unphysical negative
            temperatures (which can appear during a diverging coupled
            iteration) produce a finite, positive result rather than nan.

        Returns
        -------
        k : same type/shape as T
            Dimensional thermal conductivity (W/m-K).
        """
        T_safe = np.maximum(T, 1.0)
        mu = self.suther1 * T_safe ** (3.0 / 2.0) / (T_safe + self.suther2)
        return mu * self.cp / self.Pr

    def _sutherland_k_deriv(self, T):
        """
        Evaluate dk/dT via Sutherland's law at temperature T (K).

        Parameters
        ----------
        T : float or np.ndarray
            Temperature(s) at which to evaluate the derivative.  Values are
            floored at 1 K consistent with ``_sutherland_k``.

        Returns
        -------
        dkdT : same type/shape as T
            Derivative of dimensional thermal conductivity with respect to T (W/m-K^2).
        """
        T_safe = np.maximum(T, 1.0)
        s2 = self.suther2
        dmu_dT = (
            self.suther1
            * T_safe ** (0.5)
            * (3.0 * s2 + T_safe)
            / (2.0 * (s2 + T_safe) ** 2)
        )
        return dmu_dT * self.cp / self.Pr

    def _eckert_T_star(self, aero_temps):
        """
        Compute the Eckert reference temperature T* (K) at each aero surface node.

        The standard Eckert reference temperature is:
            T* = 0.5*(T_w + T_inf) + 0.22*(T_aw - T_inf)

        where the adiabatic-wall temperature is:
            T_aw = T_inf * (1 + r*(gamma-1)/2 * Mach_inf^2)

        and the recovery factor r is:
            r = Pr^(1/3)   (turbulent, default)
            r = Pr^(1/2)   (laminar)

        Parameters
        ----------
        aero_temps : np.ndarray
            Current wall temperatures at each aero surface node (K).

        Returns
        -------
        T_star : np.ndarray
            Eckert reference temperature at each node (K).
        """
        r = self.Pr ** (1.0 / 3.0) if self.turbulent else self.Pr**0.5
        T_aw = self.T_inf * (1.0 + r * (self.gamma - 1.0) / 2.0 * self.Mach_inf**2)
        T_star = 0.5 * (aero_temps + self.T_inf) + 0.22 * (T_aw - self.T_inf)
        return T_star

    def get_thermal_conduct(self, aero_temps):
        """
        Calculate dimensional thermal conductivity at each aero surface node.

        Dispatches to one of three strategies set at Scenario construction time via the
        ``Mach_inf`` and ``k_fixed`` arguments:

        * ``"eckert"``  (recommended for aerothermal problems): evaluates Sutherland's law at
          the Eckert reference temperature T*, which removes the positive-feedback loop that
          causes instability when k is evaluated at the wall temperature.  Requires
          ``Mach_inf`` to be set.
        * ``"fixed"``   (most conservative): returns the user-supplied constant ``k_fixed``
          broadcast to the size of ``aero_temps``.  Globally contractive but introduces a
          steady-state bias.
        * ``"wall"``    (legacy default): evaluates Sutherland's law at the current wall
          temperature.  Unstable at high Mach / low coupling frequency — use with caution.

        Parameters
        ----------
        aero_temps : np.ndarray
            Current aero surface temperatures (K).

        Returns
        -------
        k : np.ndarray
            Dimensional thermal conductivity at each node (W/m-K).
        """
        if self.k_eval_strategy == "fixed":
            return np.full_like(aero_temps, self.k_fixed)

        if self.k_eval_strategy == "eckert":
            T_star = self._eckert_T_star(aero_temps)
            return self._sutherland_k(T_star)

        # "wall" strategy — legacy behaviour with a one-time warning
        if not getattr(self, "_wall_strategy_warned", False):
            import warnings

            warnings.warn(
                f"Scenario '{self.name}': k_eval_strategy='wall' evaluates thermal "
                "conductivity at the current wall temperature. This is known to be "
                "unstable in aerothermal coupling at significant Mach numbers. "
                "Set Mach_inf when constructing the Scenario to use the stable "
                "'eckert' strategy, or supply k_fixed for a fixed-k fallback.",
                UserWarning,
                stacklevel=2,
            )
            self._wall_strategy_warned = True
        return self._sutherland_k(aero_temps)

    def get_thermal_conduct_deriv(self, aero_temps):
        """
        Calculate dk/dT_wall at each aero surface node, consistent with the active
        ``k_eval_strategy``.

        * ``"eckert"``: applies the chain rule through T*.  Since dT*/dT_w = 0.5, this
          returns ``dk/dT* * 0.5`` — exactly halving the destabilizing sensitivity
          compared with the ``"wall"`` strategy.
        * ``"fixed"``: returns zeros (k is constant, no sensitivity to wall temperature).
        * ``"wall"``: returns ``dk/dT_w`` directly from Sutherland's law (legacy).

        Parameters
        ----------
        aero_temps : np.ndarray
            Current aero surface temperatures (K).

        Returns
        -------
        dkdtA : np.ndarray
            Derivative of dimensional thermal conductivity with respect to wall
            temperature at each node (W/m-K^2).
        """
        if self.k_eval_strategy == "fixed":
            return np.zeros_like(aero_temps)

        if self.k_eval_strategy == "eckert":
            T_star = self._eckert_T_star(aero_temps)
            # chain rule: dk/dT_w = dk/dT* * dT*/dT_w,  dT*/dT_w = 0.5
            return self._sutherland_k_deriv(T_star) * 0.5

        # "wall" strategy — direct Sutherland derivative
        return self._sutherland_k_deriv(aero_temps)

    def __str__(self):
        line1 = f"Scenario (<ID> <Name>): {self.id} {self.name}"
        line2 = f"    Coupling Group: {self.group}"
        line3 = f"    Steps: {self.steps}"
        line4 = f"    Steady-state: {self.steady}"

        output = (line1, line2, line3, line4)

        return "\n".join(output)

    def print_summary(self, comm=None, filename=None):
        """
        Print a detailed summary of the scenario's step counts, coupling
        frequencies, early-stopping settings, flow reference values, and
        thermal/gas constants.

        Parameters
        ----------
        comm : MPI communicator, optional
            If provided, a barrier is inserted before and after printing and
            only rank 0 produces output.
        filename : str or path-like, optional
            If provided, the summary is written to this file (opened in write
            mode and closed after printing). If None, prints to stdout.
        """
        print_here = True
        if comm is not None:
            comm.Barrier()
            if comm.rank != 0:
                print_here = False

        if not print_here:
            if comm is not None:
                comm.Barrier()
            return

        if filename is not None:
            fp = open(filename, "w")
        else:
            fp = None

        p = lambda *args, **kw: print(*args, file=fp, **kw)

        p("--------------------------------------------------")
        p(f"| Scenario Summary: {self.name}")
        p("--------------------------------------------------")

        # --- Identity ---
        p(f"  Name               : {self.name}")
        p(f"  ID                 : {self.id}")
        p(f"  Group              : {self.group}")
        p(f"  Steady             : {self.steady}")
        p(f"  FUN3D project name : {self.fun3d_project_name}")

        # --- Forward solve steps ---
        p("")
        p("  Forward Solve")
        p("  -------------")
        p(f"  Uncoupled steps          : {self.uncoupled_steps}")
        p(f"  Coupled steps            : {self.steps}")
        p(f"  Forward coupling freq    : {self.forward_coupling_frequency}")
        total_fwd = self.steps * self.forward_coupling_frequency + self.uncoupled_steps
        p(f"  Total flow iterations    : {total_fwd}")
        if self.post_tight_forward_steps > 0:
            p(f"  Post tight-coupling steps: {self.post_tight_forward_steps}")
            p(f"  Post tight coupling freq : {self.post_forward_coupling_freq}")

        # --- Adjoint solve steps ---
        p("")
        p("  Adjoint Solve")
        p("  -------------")
        p(f"  Adjoint steps            : {self.adjoint_steps}")
        p(f"  Adjoint coupling freq    : {self.adjoint_coupling_frequency}")
        if self.adjoint_steps is not None:
            total_adj = self.adjoint_steps * self.adjoint_coupling_frequency
            p(f"  Total adjoint iterations : {total_adj}")
        if self.post_tight_adjoint_steps > 0:
            p(f"  Post tight-coupling steps: {self.post_tight_adjoint_steps}")
            p(f"  Post tight coupling freq : {self.post_adjoint_coupling_freq}")

        # --- Early stopping ---
        p("")
        p("  Early Stopping")
        p("  --------------")
        p(f"  Enabled                  : {self.early_stopping}")
        p(f"  Min forward steps        : {self.min_forward_steps}")
        p(f"  Min adjoint steps        : {self.min_adjoint_steps}")

        # --- Flow reference values ---
        p("")
        p("  Flow Reference Values")
        p("  ---------------------")
        p(f"  qinf (dyn. pressure)     : {self.qinf}")
        p(f"  flow_dt (nondim dt)      : {self.flow_dt}")
        p(f"  T_ref (struct ref temp)  : {self.T_ref} K")
        p(f"  T_inf (freestream temp)  : {self.T_inf} K")

        # --- Gas / thermal constants ---
        p("")
        p("  Gas / Thermal Constants")
        p("  -----------------------")
        p(f"  gamma                    : {self.gamma}")
        p(f"  R_specific               : {self.R_specific} J/kg-K")
        p(f"  Pr (Prandtl)             : {self.Pr}")
        p(f"  Sutherland C1            : {self.suther1} kg/m-s-K^0.5")
        p(f"  Sutherland C2            : {self.suther2} K")
        p(f"  cp                       : {self.cp:.6g} J/kg-K")
        p(f"  k_eval_strategy          : {self.k_eval_strategy}")
        if self.k_eval_strategy == "eckert":
            p(f"  Mach_inf                 : {self.Mach_inf}")
            p(f"  turbulent recovery factor: {self.turbulent}")
        elif self.k_eval_strategy == "fixed":
            p(f"  k_fixed                  : {self.k_fixed} W/m-K")

        # --- Functions ---
        p("")
        p("  Functions")
        p("  ---------")
        if self.functions:
            self._print_functions(file=fp)
        else:
            p("    (none registered)")

        # --- Variables ---
        p("")
        p("  Variables")
        p("  ---------")
        if any(self.variables[vt] for vt in self.variables):
            for vartype in self.variables:
                p(
                    f"    Variable type : {vartype}  ({len(self.variables[vartype])} vars)"
                )
                self._print_variables(vartype, file=fp)
        else:
            p("    (none registered)")

        if fp is not None:
            fp.close()

        if comm is not None:
            comm.Barrier()

        return
