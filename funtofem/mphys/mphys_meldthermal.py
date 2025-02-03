__all__ = ["MeldTempXfer", "MeldHeatXfer", "MeldThermalBuilder"]

import numpy as np
import openmdao.api as om
from funtofem import TransferScheme
from mphys import Builder, MPhysVariables

""" builder and components to wrap meld thermal to transfert temperature and
heat transfer rate between the convective and conductive analysis."""

# Set MPhys variable names
X_THERMAL0 = MPhysVariables.Thermal.Mesh.COORDINATES
X_AERO_SURFACE0 = MPhysVariables.Aerodynamics.Surface.COORDINATES_INITIAL
T_THERMAL = MPhysVariables.Thermal.TEMPERATURE
T_AERO = MPhysVariables.Aerodynamics.Surface.TEMPERATURE
Q_AERO = MPhysVariables.Aerodynamics.Surface.HEAT_FLOW
Q_THERMAL = MPhysVariables.Thermal.HeatFlow.AERODYNAMIC


class MeldTempXfer(om.ExplicitComponent):
    """
    Component to perform displacement transfer using Meld
    """

    def initialize(self):
        self.options.declare("xfer_object", recordable=False)
        self.options.declare("thermal_ndof")
        self.options.declare("thermal_nnodes")

        self.options.declare("aero_nnodes")
        self.options.declare("check_partials")

        self.meldThermal = None
        self.initialized_meld = False

        self.thermal_ndof = None
        self.thermal_nnodes = None
        self.aero_nnodes = None
        self.under_check_partials = False

    def setup(self):
        self.meldThermal = self.options["xfer_object"]

        self.thermal_ndof = self.options["thermal_ndof"]
        self.thermal_nnodes = self.options["thermal_nnodes"]
        self.aero_nnodes = self.options["aero_nnodes"]
        self.under_check_partials = self.options["check_partials"]
        aero_nnodes = self.aero_nnodes

        # initialization inputs
        self.add_input(
            X_THERMAL0,
            distributed=True,
            shape_by_conn=True,
            desc="initial thermal node coordinates",
            tags=["mphys_coordinates"],
        )
        self.add_input(
            X_AERO_SURFACE0,
            distributed=True,
            shape_by_conn=True,
            desc="initial aerodynamic surface node coordinates",
            tags=["mphys_coordinates"],
        )
        # inputs
        self.add_input(
            T_THERMAL,
            distributed=True,
            shape_by_conn=True,
            desc="thermalive node displacements",
            tags=["mphys_coupling"],
        )

        # outputs

        self.add_output(
            T_AERO,
            shape_by_conn=True,
            distributed=True,
            desc="aero surface temperatures",
            tags=["mphys_coupling"],
        )
        self.meld_initialized = False

    def compute(self, inputs, outputs):
        if not self.meld_initialized:
            x_t0 = np.array(inputs[X_THERMAL0], dtype=TransferScheme.dtype)
            x_a0 = np.array(inputs[X_AERO_SURFACE0], dtype=TransferScheme.dtype)

            self.meldThermal.setStructNodes(x_t0)
            self.meldThermal.setAeroNodes(x_a0)

            self.meldThermal.initialize()
            self.meld_initialized = True

        T_t = np.array(inputs[T_THERMAL], dtype=TransferScheme.dtype)
        T_a = np.array(outputs[T_AERO], dtype=TransferScheme.dtype)

        self.meldThermal.transferTemp(T_t, T_a)
        outputs[T_AERO] = T_a

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        The explicit component is defined as:
            u_a = g(u_s,x_a0,x_s0)
        The Meld residual is defined as:
            D = u_a - g(u_s,x_a0,x_s0)
        So explicit partials below for u_a are negative partials of D
        """

        meld = self.meldThermal
        if mode == "fwd":
            if T_AERO in d_outputs:
                if T_THERMAL in d_inputs:
                    d_T_thermal = np.array(
                        d_inputs[T_THERMAL], dtype=TransferScheme.dtype
                    )

                    prod = np.zeros(d_outputs[T_AERO].size, dtype=TransferScheme.dtype)

                    meld.applydTdtS(d_T_thermal, prod)
                    d_outputs[T_AERO] += np.array(prod, dtype=float)

                if X_AERO_SURFACE0 in d_inputs:
                    if self.under_check_partials:
                        pass
                    else:
                        raise ValueError("forward mode requested but not implemented")

                if X_THERMAL0 in d_inputs:
                    if self.under_check_partials:
                        pass
                    else:
                        raise ValueError("forward mode requested but not implemented")

        if mode == "rev":
            if T_AERO in d_outputs:
                dT_aero = np.array(d_outputs[T_AERO], dtype=TransferScheme.dtype)
                if T_THERMAL in d_inputs:
                    # dT_aero/dT_THERMAL^T * psi = - dD/dT_THERMAL^T psi

                    prod = np.zeros(
                        d_inputs[T_THERMAL].size, dtype=TransferScheme.dtype
                    )

                    meld.applydTdtSTrans(dT_aero, prod)

                    d_inputs[T_THERMAL] -= np.array(prod, dtype=np.float64)


class MeldHeatXfer(om.ExplicitComponent):
    """
    Component to perform load transfers using Meld
    """

    def initialize(self):
        self.options.declare("xfer_object", recordable=False)
        self.options.declare("thermal_ndof")
        self.options.declare("thermal_nnodes")

        self.options.declare("aero_nnodes")
        self.options.declare("check_partials")

        self.meldThermal = None
        self.initialized_meld = False

        self.thermal_ndof = None
        self.thermal_nnodes = None
        self.aero_nnodes = None
        self.under_check_partials = False

    def setup(self):
        # get the transfer scheme object
        self.meldThermal = self.options["xfer_object"]

        self.thermal_ndof = self.options["thermal_ndof"]
        self.thermal_nnodes = self.options["thermal_nnodes"]
        self.aero_nnodes = self.options["aero_nnodes"]
        self.under_check_partials = self.options["check_partials"]

        # initialization inputs
        self.add_input(
            X_THERMAL0,
            distributed=True,
            shape_by_conn=True,
            desc="initial thermal node coordinates",
            tags=["mphys_coordinates"],
        )
        self.add_input(
            X_AERO_SURFACE0,
            distributed=True,
            shape_by_conn=True,
            desc="initial aerodynamic surface node coordinates",
            tags=["mphys_coordinates"],
        )

        # inputs
        self.add_input(
            Q_AERO,
            distributed=True,
            shape_by_conn=True,
            desc="initial aero heat transfer rate",
            tags=["mphys_coupling"],
        )

        # outputs
        self.add_output(
            Q_THERMAL,
            distributed=True,
            shape_by_conn=True,
            desc="heat transfer rate on the thermalion mesh at the interface",
            tags=["mphys_coupling"],
        )

        self.meld_initialized = False

    def compute(self, inputs, outputs):
        if not self.meld_initialized:
            x_t0 = np.array(inputs[X_THERMAL0], dtype=TransferScheme.dtype)
            x_a0 = np.array(inputs[X_AERO_SURFACE0], dtype=TransferScheme.dtype)

            self.meldThermal.setStructNodes(x_t0)
            self.meldThermal.setAeroNodes(x_a0)

            self.meldThermal.initialize()
            self.meld_initialized = True

        heaT_AERO = np.array(inputs[Q_AERO], dtype=TransferScheme.dtype)
        heaT_THERMAL = np.array(outputs[Q_THERMAL], dtype=TransferScheme.dtype)
        self.meldThermal.transferFlux(heaT_AERO, heaT_THERMAL)
        outputs[Q_THERMAL] = heaT_THERMAL

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        The explicit component is defined as:
            u_a = g(u_s,x_a0,x_s0)
        The Meld residual is defined as:
            D = u_a - g(u_s,x_a0,x_s0)
        So explicit partials below for u_a are negative partials of D
        """

        meld = self.meldThermal
        if mode == "fwd":
            if Q_THERMAL in d_outputs:
                if Q_AERO in d_inputs:
                    d_q_aero = np.array(d_inputs[Q_AERO], dtype=TransferScheme.dtype)

                    prod = np.zeros(
                        d_outputs[Q_THERMAL].size, dtype=TransferScheme.dtype
                    )

                    meld.applydQdqA(d_q_aero, prod)
                    d_outputs[Q_THERMAL] += np.array(prod, dtype=np.float64)

                if X_AERO_SURFACE0 in d_inputs:
                    if self.under_check_partials:
                        pass
                    else:
                        raise ValueError("forward mode requested but not implemented")

                if X_THERMAL0 in d_inputs:
                    if self.under_check_partials:
                        pass
                    else:
                        raise ValueError("forward mode requested but not implemented")

        if mode == "rev":
            if Q_THERMAL in d_outputs:
                dq_thermal = np.array(d_outputs[Q_THERMAL], dtype=TransferScheme.dtype)
                if Q_AERO in d_inputs:
                    prod = np.zeros(d_inputs[Q_AERO].size, dtype=TransferScheme.dtype)

                    meld.applydQdqATrans(dq_thermal, prod)

                    d_inputs[Q_AERO] -= np.array(prod, dtype=np.float64)


class MeldThermalBuilder(Builder):
    def __init__(
        self,
        aero_builder,
        thermal_builder,
        isym=-1,
        n=200,
        beta=0.5,
        check_partials=False,
    ):
        # super(MeldThermalBuilder, self).__init__(options)
        # TODO we can move the aero and thermal builder to init_xfer_object call so that user does not need to worry about this
        self.aero_builder = aero_builder
        self.thermal_builder = thermal_builder
        self.isym = isym
        self.n = n
        self.beta = beta
        self.under_check_partials = check_partials

    def initialize(self, comm):
        self.nnodes_aero = self.aero_builder.get_number_of_nodes()
        self.nnodes_thermal = self.thermal_builder.get_number_of_nodes()
        self.ndof_thermal = self.thermal_builder.get_ndof()

        self.meldthermal = TransferScheme.pyMELDThermal(
            comm,
            comm,
            0,
            comm,
            0,
            self.isym,
            self.n,
            self.beta,
        )

    def get_coupling_group_subsystem(self, scenario_name=None):
        heat_xfer = MeldHeatXfer(
            xfer_object=self.meldthermal,
            thermal_ndof=self.ndof_thermal,
            thermal_nnodes=self.nnodes_thermal,
            aero_nnodes=self.nnodes_aero,
            check_partials=self.under_check_partials,
        )

        temp_xfer = MeldTempXfer(
            xfer_object=self.meldthermal,
            thermal_ndof=self.ndof_thermal,
            thermal_nnodes=self.nnodes_thermal,
            aero_nnodes=self.nnodes_aero,
            check_partials=self.under_check_partials,
        )

        return heat_xfer, temp_xfer
