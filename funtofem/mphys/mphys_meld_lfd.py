#!/usr/bin/env python

__all__ = ["ModeTransfer", "MeldLfdBuilder"]

import numpy as np
import openmdao.api as om
from .mphys_meld import MeldBuilder

from funtofem import TransferScheme

# Set MPhys variable names
X_STRUCT0 = MPhysVariables.Structures.COORDINATES
X_AERO0 = MPhysVariables.Aerodynamics.Surface.COORDINATES_INITIAL


class ModeTransfer(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("nmodes")
        self.options.declare("nnodes_struct")
        self.options.declare("ndof_struct")
        self.options.declare("nnodes_aero")
        self.options.declare("meld")
        self.options.declare(
            "use_ref_coordinates",
            types=bool,
            desc="Use separate aero and struct reference coordinates for transfer scheme initialization (same variable name with '_ref' appended)",
        )

        self.first_pass = True

    def setup(self):
        # self.set_check_partial_options(wrt='*',method='cs',directional=True)

        self.add_input(
            X_STRUCT0,
            shape_by_conn=True,
            distributed=True,
            tags=["mphys_coordinates"],
        )
        self.add_input(
            "mode_shapes_struct",
            shape_by_conn=True,
            distributed=True,
            tags="mphys_coupling",
        )
        self.add_input(
            X_AERO0, shape_by_conn=True, distributed=True, tags=["mphys_coordinates"]
        )
        if self.options["use_ref_coordinates"]:
            self.add_input(
                X_STRUCT0 + "_ref",
                shape_by_conn=True,
                distributed=True,
                desc="baseline structural node coordinates with which to initialize MELD",
                tags=["mphys_coordinates"],
            )
            self.add_input(
                X_AERO0 + "_ref",
                shape_by_conn=True,
                distributed=True,
                desc="baseline aero surface node coordinates with which to initialize MELD",
                tags=["mphys_coordinates"],
            )

        nmodes = self.options["nmodes"]
        self.nnodes_aero = self.options["nnodes_aero"]
        self.nnodes_struct = self.options["nnodes_struct"]
        self.ndof_struct = self.options["ndof_struct"]

        aero_mode_size = (self.nnodes_aero * 3, nmodes)
        self.add_output(
            "mode_shapes_aero",
            shape=aero_mode_size,
            distributed=True,
            tags=["mphys_coupling"],
        )

    def _initialize_xfer(self, inputs, meld):
        if self.options["use_ref_coordinates"]:
            aero_X = np.array(inputs[X_AERO0 + "_ref"], dtype=TransferScheme.dtype)
            struct_X = np.array(inputs[X_STRUCT0 + "_ref"], dtype=TransferScheme.dtype)
        else:
            aero_X = np.array(inputs[X_AERO0], dtype=TransferScheme.dtype)
            struct_X = np.array(inputs[X_STRUCT0], dtype=TransferScheme.dtype)
        meld.setStructNodes(struct_X)
        meld.setAeroNodes(aero_X)
        meld.initialize()

    def compute(self, inputs, outputs):
        meld = self.options["meld"]
        nmodes = self.options["nmodes"]
        aero_X = np.array(inputs[X_AERO0], dtype=TransferScheme.dtype)
        struct_X = np.array(inputs[X_STRUCT0], dtype=TransferScheme.dtype)

        aero_modes = np.zeros((aero_X.size, nmodes), dtype=TransferScheme.dtype)
        struct_modes = inputs["mode_shapes_struct"].reshape((-1, nmodes))

        meld.setAeroNodes(aero_X)
        meld.setStructNodes(struct_X)

        if self.first_pass:
            self._initialize_xfer(inputs, meld)
            self.first_pass = False

        struct_mode = np.zeros(self.nnodes_struct * 3, dtype=TransferScheme.dtype)
        for mode in range(nmodes):
            struct_mode[0::3] = struct_modes[0 :: self.ndof_struct, mode]
            struct_mode[1::3] = struct_modes[1 :: self.ndof_struct, mode]
            struct_mode[2::3] = struct_modes[2 :: self.ndof_struct, mode]
            aero_mode = np.zeros_like(aero_X)
            meld.transferDisps(struct_mode, aero_mode)
            aero_modes[:, mode] = aero_mode

        outputs["mode_shapes_aero"] = aero_modes.copy()

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        meld = self.options["meld"]
        nmodes = self.options["nmodes"]
        x_s0 = np.array(inputs[X_STRUCT0], dtype=TransferScheme.dtype)
        x_a0 = np.array(inputs[X_AERO0], dtype=TransferScheme.dtype)

        meld.setStructNodes(x_s0)
        meld.setAeroNodes(x_a0)

        for imode in range(nmodes):
            u_s = np.zeros(self.nnodes_struct * 3, dtype=TransferScheme.dtype)
            for i in range(3):
                u_s[i::3] = inputs["mode_shapes_struct"][i :: self.ndof_struct, imode]
            u_a = np.zeros(self.nnodes_aero * 3, dtype=TransferScheme.dtype)
            meld.transferDisps(u_s, u_a)
            if mode == "fwd":
                if "mode_shapes_aero" in d_outputs:
                    if "mode_shapes_struct" in d_inputs:
                        d_in = np.zeros(
                            self.nnodes_struct * 3, dtype=TransferScheme.dtype
                        )
                        for i in range(3):
                            d_in[i::3] = d_inputs["mode_shapes_struct"][
                                i :: self.ndof_struct, imode
                            ]
                        prod = np.zeros(
                            self.nnodes_aero * 3, dtype=TransferScheme.dtype
                        )
                        meld.applydDduS(d_in, prod)
                        d_outputs["mode_shapes_aero"][:, imode] -= np.array(
                            prod, dtype=float
                        )
            if mode == "rev":
                if "mode_shapes_aero" in d_outputs:
                    du_a = np.array(
                        d_outputs["mode_shapes_aero"][:, imode],
                        dtype=TransferScheme.dtype,
                    )
                    if "mode_shapes_struct" in d_inputs:
                        # du_a/du_s^T * psi = - dD/du_s^T psi
                        prod = np.zeros(
                            self.nnodes_struct * 3, dtype=TransferScheme.dtype
                        )
                        meld.applydDduSTrans(du_a, prod)
                        for i in range(3):
                            d_inputs["mode_shapes_struct"][
                                i :: self.ndof_struct, imode
                            ] -= np.array(prod[i::3], dtype=np.float64)

                    # du_a/dx_a0^T * psi = - psi^T * dD/dx_a0 in F2F terminology
                    if X_AERO0 in d_inputs:
                        prod = np.zeros(
                            d_inputs[X_AERO0].size, dtype=TransferScheme.dtype
                        )
                        meld.applydDdxA0(du_a, prod)
                        d_inputs[X_AERO0] -= np.array(prod, dtype=float)

                    if X_STRUCT0 in d_inputs:
                        prod = np.zeros(
                            self.nnodes_struct * 3, dtype=TransferScheme.dtype
                        )
                        meld.applydDdxS0(du_a, prod)
                        d_inputs[X_STRUCT0] -= np.array(prod, dtype=float)


class MeldLfdBuilder(MeldBuilder):
    def __init__(
        self,
        aero_builder,
        struct_builder,
        nmodes,
        isym=-1,
        n=200,
        beta=0.5,
        check_partials=False,
        use_ref_coordinates=False,
    ):
        self.nmodes = nmodes
        super().__init__(aero_builder, struct_builder, isym, n, beta, check_partials, False, None, use_ref_coordinates)

    def get_post_coupling_subsystem(self):
        return ModeTransfer(
            nmodes=self.nmodes,
            nnodes_struct=self.nnodes_struct,
            ndof_struct=self.ndof_struct,
            nnodes_aero=self.nnodes_aero,
            meld=self.meld,
            use_ref_coordinates=self.use_ref_coordinates,
        )
