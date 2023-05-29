__all__ = ["MeldTempXfer", "MeldHeatXfer", "MeldThermalBuilder"]

import numpy as np
import openmdao.api as om
from pyfuntofem import TransferScheme
from mphys import Builder

""" builder and components to wrap meld thermal to transfert temperature and
heat transfer rate between the convective and conductive analysis."""


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
        self.check_partials = False

    def setup(self):
        self.meldThermal = self.options["xfer_object"]

        self.thermal_ndof = self.options["thermal_ndof"]
        self.thermal_nnodes = self.options["thermal_nnodes"]
        self.aero_nnodes = self.options["aero_nnodes"]
        self.check_partials = self.options["check_partials"]
        aero_nnodes = self.aero_nnodes

        # intialization inputs
        self.add_input(
            "x_thermal_surface0",
            distributed=True,
            shape_by_conn=True,
            desc="initial thermal node coordinates",
            tags=["mphys_coordinates"],
        )
        self.add_input(
            "x_aero_surface0",
            distributed=True,
            shape_by_conn=True,
            desc="initial aerodynamic surface node coordinates",
            tags=["mphys_coordinates"],
        )
        # inputs
        self.add_input(
            "T_conduct",
            distributed=True,
            shape_by_conn=True,
            desc="thermalive node displacements",
            tags=["mphys_coupling"],
        )

        # outputs

        self.add_output(
            "T_convect",
            shape_by_conn=True,
            distributed=True,
            desc="aero surface temperatures",
            tags=["mphys_coupling"],
        )
        self.meld_initialized = False

    def compute(self, inputs, outputs):
        if not self.meld_initialized:
            x_thermal_surface0 = np.array(
                inputs["x_thermal_surface0"], dtype=TransferScheme.dtype
            )
            x_aero_surface0 = np.array(
                inputs["x_aero_surface0"], dtype=TransferScheme.dtype
            )

            self.meldThermal.setStructNodes(x_thermal_surface0)
            self.meldThermal.setAeroNodes(x_aero_surface0)

            self.meldThermal.initialize()
            self.meld_initialized = True

        T_conduct = np.array(inputs["T_conduct"], dtype=TransferScheme.dtype)
        T_convect = np.array(outputs["T_convect"], dtype=TransferScheme.dtype)

        self.meldThermal.transferTemp(T_conduct, T_convect)
        outputs["T_convect"] = T_convect

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
            if "T_convect" in d_outputs:
                if "T_conduct" in d_inputs:
                    d_T_conduct = np.array(
                        d_inputs["T_conduct"], dtype=TransferScheme.dtype
                    )

                    prod = np.zeros(
                        d_outputs["T_convect"].size, dtype=TransferScheme.dtype
                    )

                    meld.applydTdtS(d_T_conduct, prod)
                    d_outputs["T_convect"] += np.array(prod, dtype=float)

                if "x_aero_surface0" in d_inputs:
                    if self.check_partials:
                        pass
                    else:
                        raise ValueError("forward mode requested but not implemented")

                if "x_thermal_surface0" in d_inputs:
                    if self.check_partials:
                        pass
                    else:
                        raise ValueError("forward mode requested but not implemented")

        if mode == "rev":
            if "T_convect" in d_outputs:
                dT_convect = np.array(
                    d_outputs["T_convect"], dtype=TransferScheme.dtype
                )
                if "T_conduct" in d_inputs:
                    # dT_convect/dT_conduct^T * psi = - dD/dT_conduct^T psi

                    prod = np.zeros(
                        d_inputs["T_conduct"].size, dtype=TransferScheme.dtype
                    )

                    meld.applydTdtSTrans(dT_convect, prod)

                    d_inputs["T_conduct"] -= np.array(prod, dtype=np.float64)


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
        self.check_partials = False

    def setup(self):
        # get the transfer scheme object
        self.meldThermal = self.options["xfer_object"]

        self.thermal_ndof = self.options["thermal_ndof"]
        self.thermal_nnodes = self.options["thermal_nnodes"]
        self.aero_nnodes = self.options["aero_nnodes"]
        self.check_partials = self.options["check_partials"]

        # initialization inputs
        self.add_input(
            "x_thermal_surface0",
            distributed=True,
            shape_by_conn=True,
            desc="initial thermal node coordinates",
            tags=["mphys_coordinates"],
        )
        self.add_input(
            "x_aero_surface0",
            distributed=True,
            shape_by_conn=True,
            desc="initial aerodynamic surface node coordinates",
            tags=["mphys_coordinates"],
        )

        # inputs
        self.add_input(
            "q_convect",
            distributed=True,
            shape_by_conn=True,
            desc="initial aero heat transfer rate",
            tags=["mphys_coupling"],
        )

        # outputs
        self.add_output(
            "q_conduct",
            distributed=True,
            shape_by_conn=True,
            desc="heat transfer rate on the thermalion mesh at the interface",
            tags=["mphys_coupling"],
        )

        self.meld_initialized = False

    def compute(self, inputs, outputs):
        if not self.meld_initialized:
            x_thermal_surface0 = np.array(
                inputs["x_thermal_surface0"], dtype=TransferScheme.dtype
            )
            x_aero_surface0 = np.array(
                inputs["x_aero_surface0"], dtype=TransferScheme.dtype
            )

            self.meldThermal.setStructNodes(x_thermal_surface0)
            self.meldThermal.setAeroNodes(x_aero_surface0)

            self.meldThermal.initialize()
            self.meld_initialized = True

        heat_convect = np.array(inputs["q_convect"], dtype=TransferScheme.dtype)
        heat_conduct = np.array(outputs["q_conduct"], dtype=TransferScheme.dtype)
        self.meldThermal.transferFlux(heat_convect, heat_conduct)
        outputs["q_conduct"] = heat_conduct

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
            if "q_conduct" in d_outputs:
                if "q_convect" in d_inputs:
                    d_q_convect = np.array(
                        d_inputs["q_convect"], dtype=TransferScheme.dtype
                    )

                    prod = np.zeros(
                        d_outputs["q_conduct"].size, dtype=TransferScheme.dtype
                    )

                    meld.applydQdqA(d_q_convect, prod)
                    d_outputs["q_conduct"] += np.array(prod, dtype=np.float64)

                if "x_aero_surface0" in d_inputs:
                    if self.check_partials:
                        pass
                    else:
                        raise ValueError("forward mode requested but not implemented")

                if "x_thermal_surface0" in d_inputs:
                    if self.check_partials:
                        pass
                    else:
                        raise ValueError("forward mode requested but not implemented")

        if mode == "rev":
            if "q_conduct" in d_outputs:
                dq_conduct = np.array(
                    d_outputs["q_conduct"], dtype=TransferScheme.dtype
                )
                if "q_convect" in d_inputs:
                    prod = np.zeros(
                        d_inputs["q_convect"].size, dtype=TransferScheme.dtype
                    )

                    meld.applydQdqATrans(dq_conduct, prod)

                    d_inputs["q_convect"] -= np.array(prod, dtype=np.float64)


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
        self.check_partials = check_partials

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
            check_partials=self.check_partials,
        )

        temp_xfer = MeldTempXfer(
            xfer_object=self.meldthermal,
            thermal_ndof=self.ndof_thermal,
            thermal_nnodes=self.nnodes_thermal,
            aero_nnodes=self.nnodes_aero,
            check_partials=self.check_partials,
        )

        return heat_xfer, temp_xfer
