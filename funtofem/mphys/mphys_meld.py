__all__ = ["MeldLoadXfer", "MeldDispXfer", "MeldBuilder"]

import numpy as np
import openmdao.api as om
from mphys import Builder, MPhysVariables

from funtofem import TransferScheme

# Set MPhys variable names
X_STRUCT0 = MPhysVariables.Structures.COORDINATES
X_AERO0 = MPhysVariables.Aerodynamics.Surface.COORDINATES_INITIAL
U_STRUCT = MPhysVariables.Structures.DISPLACEMENTS
U_AERO = MPhysVariables.Aerodynamics.Surface.DISPLACEMENTS
F_AERO = MPhysVariables.Aerodynamics.Surface.LOADS
F_STRUCT = MPhysVariables.Structures.Loads.AERODYNAMIC


class MeldDispXfer(om.ExplicitComponent):
    """
    Component to perform displacement transfer using MELD
    """

    def initialize(self):
        self.options.declare("struct_ndof")
        self.options.declare("struct_nnodes")
        self.options.declare("aero_nnodes")
        self.options.declare("check_partials")
        self.options.declare("bodies", recordable=False)

        self.struct_ndof = None
        self.struct_nnodes = None
        self.aero_nnodes = None
        self.under_check_partials = False

    def setup(self):
        self.struct_ndof = self.options["struct_ndof"]
        self.struct_nnodes = self.options["struct_nnodes"]
        self.aero_nnodes = self.options["aero_nnodes"]
        self.under_check_partials = self.options["check_partials"]
        self.bodies = self.options["bodies"]

        # self.set_check_partial_options(wrt='*',method='cs',directional=True)

        # inputs
        self.add_input(
            X_STRUCT0,
            shape_by_conn=True,
            distributed=True,
            desc="initial structural node coordinates",
            tags=["mphys_coordinates"],
        )
        self.add_input(
            X_AERO0,
            shape_by_conn=True,
            distributed=True,
            desc="initial aero surface node coordinates",
            tags=["mphys_coordinates"],
        )
        self.add_input(
            U_STRUCT,
            shape_by_conn=True,
            distributed=True,
            desc="structural node displacements",
            tags=["mphys_coupling"],
        )

        # outputs
        self.add_output(
            U_AERO,
            shape=self.aero_nnodes * 3,
            distributed=True,
            val=np.zeros(self.aero_nnodes * 3),
            desc="aerodynamic surface displacements",
            tags=["mphys_coupling"],
        )

        # partials
        # self.declare_partials(U_AERO,[X_STRUCT0,X_AERO0,U_STRUCT])

    def compute(self, inputs, outputs):
        for body in self.bodies:
            x_s0 = np.array(
                inputs[X_STRUCT0][body.struct_coord_indices],
                dtype=TransferScheme.dtype,
            )
            x_a0 = np.array(
                inputs[X_AERO0][body.aero_coord_indices], dtype=TransferScheme.dtype
            )
            u_a = np.array(
                outputs[U_AERO][body.aero_coord_indices], dtype=TransferScheme.dtype
            )

            u_s = np.zeros(len(body.struct_coord_indices), dtype=TransferScheme.dtype)
            for i in range(3):
                u_s[i::3] = inputs[U_STRUCT][
                    body.struct_dof_indices[i :: self.struct_ndof]
                ]

            body.meld.setStructNodes(x_s0)
            body.meld.setAeroNodes(x_a0)

            if not body.initialized_meld:
                body.meld.initialize()
                body.initialized_meld = True

            body.meld.transferDisps(u_s, u_a)

            outputs[U_AERO][body.aero_coord_indices] = u_a

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        The explicit component is defined as:
            u_a = g(u_s,x_a0,x_s0)
        The MELD residual is defined as:
            D = u_a - g(u_s,x_a0,x_s0)
        So explicit partials below for u_a are negative partials of D
        """

        for body in self.bodies:
            if self.under_check_partials:
                x_s0 = np.array(
                    inputs[X_STRUCT0][body.struct_coord_indices],
                    dtype=TransferScheme.dtype,
                )
                x_a0 = np.array(
                    inputs[X_AERO0][body.aero_coord_indices],
                    dtype=TransferScheme.dtype,
                )
                body.meld.setStructNodes(x_s0)
                body.meld.setAeroNodes(x_a0)
            u_s = np.zeros(len(body.struct_coord_indices), dtype=TransferScheme.dtype)
            for i in range(3):
                u_s[i::3] = inputs[U_STRUCT][
                    body.struct_dof_indices[i :: self.struct_ndof]
                ]
            u_a = np.zeros(len(body.aero_coord_indices), dtype=TransferScheme.dtype)
            body.meld.transferDisps(u_s, u_a)

            if mode == "fwd":
                if U_AERO in d_outputs:
                    if U_STRUCT in d_inputs:
                        d_in = np.zeros(
                            len(body.struct_coord_indices), dtype=TransferScheme.dtype
                        )
                        for i in range(3):
                            d_in[i::3] = d_inputs[U_STRUCT][
                                body.struct_dof_indices[i :: self.struct_ndof]
                            ]
                        prod = np.zeros(
                            len(body.aero_coord_indices), dtype=TransferScheme.dtype
                        )
                        body.meld.applydDduS(d_in, prod)
                        d_outputs[U_AERO][body.aero_coord_indices] -= np.array(
                            prod, dtype=float
                        )

                    if X_AERO0 in d_inputs:
                        if self.under_check_partials:
                            pass
                        else:
                            raise ValueError(
                                "MELD forward mode requested but not implemented"
                            )

                    if X_STRUCT0 in d_inputs:
                        if self.under_check_partials:
                            pass
                        else:
                            raise ValueError(
                                "MELD forward mode requested but not implemented"
                            )

            if mode == "rev":
                if U_AERO in d_outputs:
                    du_a = np.array(
                        d_outputs[U_AERO][body.aero_coord_indices],
                        dtype=TransferScheme.dtype,
                    )
                    if U_STRUCT in d_inputs:
                        # du_a/du_s^T * psi = - dD/du_s^T psi
                        prod = np.zeros(
                            len(body.struct_coord_indices), dtype=TransferScheme.dtype
                        )
                        body.meld.applydDduSTrans(du_a, prod)
                        for i in range(3):
                            d_inputs[U_STRUCT][
                                body.struct_dof_indices[i :: self.struct_ndof]
                            ] -= np.array(prod[i::3], dtype=np.float64)

                    # du_a/dx_a0^T * psi = - psi^T * dD/dx_a0 in F2F terminology
                    if X_AERO0 in d_inputs:
                        prod = np.zeros(
                            d_inputs[X_AERO0][body.aero_coord_indices].size,
                            dtype=TransferScheme.dtype,
                        )
                        body.meld.applydDdxA0(du_a, prod)
                        d_inputs[X_AERO0][body.aero_coord_indices] -= np.array(
                            prod, dtype=float
                        )

                    if X_STRUCT0 in d_inputs:
                        prod = np.zeros(
                            len(body.struct_coord_indices), dtype=TransferScheme.dtype
                        )
                        body.meld.applydDdxS0(du_a, prod)
                        d_inputs[X_STRUCT0][body.struct_coord_indices] -= np.array(
                            prod, dtype=float
                        )


class MeldLoadXfer(om.ExplicitComponent):
    """
    Component to perform load transfers using MELD
    """

    def initialize(self):
        self.options.declare("struct_ndof")
        self.options.declare("struct_nnodes")
        self.options.declare("aero_nnodes")
        self.options.declare("check_partials")
        self.options.declare("bodies", recordable=False)

        self.struct_ndof = None
        self.struct_nnodes = None
        self.aero_nnodes = None
        self.under_check_partials = False

    def setup(self):
        self.struct_ndof = self.options["struct_ndof"]
        self.struct_nnodes = self.options["struct_nnodes"]
        self.aero_nnodes = self.options["aero_nnodes"]
        self.under_check_partials = self.options["check_partials"]
        self.bodies = self.options["bodies"]

        # self.set_check_partial_options(wrt='*',method='cs',directional=True)

        struct_ndof = self.struct_ndof
        struct_nnodes = self.struct_nnodes

        # inputs
        self.add_input(
            X_STRUCT0,
            shape_by_conn=True,
            distributed=True,
            desc="initial structural node coordinates",
            tags=["mphys_coordinates"],
        )
        self.add_input(
            X_AERO0,
            shape_by_conn=True,
            distributed=True,
            desc="initial aero surface node coordinates",
            tags=["mphys_coordinates"],
        )
        self.add_input(
            U_STRUCT,
            shape_by_conn=True,
            distributed=True,
            desc="structural node displacements",
            tags=["mphys_coupling"],
        )
        self.add_input(
            F_AERO,
            shape_by_conn=True,
            distributed=True,
            desc="aerodynamic force vector",
            tags=["mphys_coupling"],
        )

        # outputs
        self.add_output(
            F_STRUCT,
            shape=struct_nnodes * struct_ndof,
            distributed=True,
            desc="structural force vector",
            tags=["mphys_coupling"],
        )

        # partials
        # self.declare_partials('F_STRUCT',['X_STRUCT0','X_AERO0','U_STRUCT','F_AERO'])

    def compute(self, inputs, outputs):
        outputs[F_STRUCT][:] = 0.0
        for body in self.bodies:
            if self.under_check_partials:
                x_s0 = np.array(
                    inputs[X_STRUCT0][body.struct_coord_indices],
                    dtype=TransferScheme.dtype,
                )
                x_a0 = np.array(
                    inputs[X_AERO0][body.aero_coord_indices],
                    dtype=TransferScheme.dtype,
                )
                body.meld.setStructNodes(x_s0)
                body.meld.setAeroNodes(x_a0)
            f_a = np.array(
                inputs[F_AERO][body.aero_coord_indices], dtype=TransferScheme.dtype
            )
            f_s = np.zeros(len(body.struct_coord_indices), dtype=TransferScheme.dtype)

            u_s = np.zeros(len(body.struct_coord_indices), dtype=TransferScheme.dtype)
            for i in range(3):
                u_s[i::3] = inputs[U_STRUCT][
                    body.struct_dof_indices[i :: self.struct_ndof]
                ]
            u_a = np.zeros(
                inputs[F_AERO][body.aero_coord_indices].size,
                dtype=TransferScheme.dtype,
            )
            body.meld.transferDisps(u_s, u_a)

            body.meld.transferLoads(f_a, f_s)

            for i in range(3):
                outputs[F_STRUCT][body.struct_dof_indices[i :: self.struct_ndof]] = f_s[
                    i::3
                ]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        The explicit component is defined as:
            f_s = g(f_a,u_s,x_a0,x_s0)
        The MELD internal residual is defined as:
            L = f_s - g(f_a,u_s,x_a0,x_s0)
        So explicit partials below for f_s are negative partials of L
        """

        for body in self.bodies:
            if self.under_check_partials:
                x_s0 = np.array(
                    inputs[X_STRUCT0][body.struct_coord_indices],
                    dtype=TransferScheme.dtype,
                )
                x_a0 = np.array(
                    inputs[X_AERO0][body.aero_coord_indices],
                    dtype=TransferScheme.dtype,
                )
                body.meld.setStructNodes(x_s0)
                body.meld.setAeroNodes(x_a0)
            f_a = np.array(
                inputs[F_AERO][body.aero_coord_indices], dtype=TransferScheme.dtype
            )
            f_s = np.zeros(len(body.struct_coord_indices), dtype=TransferScheme.dtype)

            u_s = np.zeros(len(body.struct_coord_indices), dtype=TransferScheme.dtype)
            for i in range(3):
                u_s[i::3] = inputs[U_STRUCT][
                    body.struct_dof_indices[i :: self.struct_ndof]
                ]
            u_a = np.zeros(
                inputs[F_AERO][body.aero_coord_indices].size,
                dtype=TransferScheme.dtype,
            )
            body.meld.transferDisps(u_s, u_a)
            body.meld.transferLoads(f_a, f_s)

            if mode == "fwd":
                if F_STRUCT in d_outputs:
                    if U_STRUCT in d_inputs:
                        d_in = np.zeros(
                            len(body.struct_coord_indices), dtype=TransferScheme.dtype
                        )
                        for i in range(3):
                            d_in[i::3] = d_inputs[U_STRUCT][
                                body.struct_dof_indices[i :: self.struct_ndof]
                            ]
                        prod = np.zeros(
                            len(body.struct_coord_indices), dtype=TransferScheme.dtype
                        )
                        body.meld.applydLduS(d_in, prod)
                        for i in range(3):
                            d_outputs[F_STRUCT][
                                body.struct_dof_indices[i :: self.struct_ndof]
                            ] -= np.array(prod[i::3], dtype=float)

                    if F_AERO in d_inputs:
                        # df_s/df_a psi = - dL/df_a * psi = -dD/du_s^T * psi
                        prod = np.zeros(
                            len(body.struct_coord_indices), dtype=TransferScheme.dtype
                        )
                        df_a = np.array(
                            d_inputs[F_AERO][body.aero_coord_indices],
                            dtype=TransferScheme.dtype,
                        )
                        body.meld.applydDduSTrans(df_a, prod)
                        for i in range(3):
                            d_outputs[F_STRUCT][
                                body.struct_dof_indices[i :: self.struct_ndof]
                            ] -= np.array(prod[i::3], dtype=float)

                    if X_AERO0 in d_inputs:
                        if self.under_check_partials:
                            pass
                        else:
                            raise ValueError(
                                "forward mode requested but not implemented"
                            )

                    if X_STRUCT0 in d_inputs:
                        if self.under_check_partials:
                            pass
                        else:
                            raise ValueError(
                                "forward mode requested but not implemented"
                            )

            if mode == "rev":
                if F_STRUCT in d_outputs:
                    d_out = np.zeros(
                        len(body.struct_coord_indices), dtype=TransferScheme.dtype
                    )
                    for i in range(3):
                        d_out[i::3] = d_outputs[F_STRUCT][
                            body.struct_dof_indices[i :: self.struct_ndof]
                        ]

                    if U_STRUCT in d_inputs:
                        d_in = np.zeros(
                            len(body.struct_coord_indices), dtype=TransferScheme.dtype
                        )
                        # df_s/du_s^T * psi = - dL/du_s^T * psi
                        body.meld.applydLduSTrans(d_out, d_in)

                        for i in range(3):
                            d_inputs[U_STRUCT][
                                body.struct_dof_indices[i :: self.struct_ndof]
                            ] -= np.array(d_in[i::3], dtype=float)

                    if F_AERO in d_inputs:
                        # df_s/df_a^T psi = - dL/df_a^T * psi = -dD/du_s * psi
                        prod = np.zeros(
                            len(body.aero_coord_indices), dtype=TransferScheme.dtype
                        )
                        body.meld.applydDduS(d_out, prod)
                        d_inputs[F_AERO][body.aero_coord_indices] -= np.array(
                            prod, dtype=float
                        )

                    if X_AERO0 in d_inputs:
                        # df_s/dx_a0^T * psi = - psi^T * dL/dx_a0 in F2F terminology
                        prod = np.zeros(
                            len(body.aero_coord_indices), dtype=TransferScheme.dtype
                        )
                        body.meld.applydLdxA0(d_out, prod)
                        d_inputs[X_AERO0][body.aero_coord_indices] -= np.array(
                            prod, dtype=float
                        )

                    if X_STRUCT0 in d_inputs:
                        # df_s/dx_s0^T * psi = - psi^T * dL/dx_s0 in F2F terminology
                        prod = np.zeros(
                            len(body.struct_coord_indices), dtype=TransferScheme.dtype
                        )
                        body.meld.applydLdxS0(d_out, prod)
                        d_inputs[X_STRUCT0][body.struct_coord_indices] -= np.array(
                            prod, dtype=float
                        )


class MeldBodyInstance:
    """
    Class that helps split OpenMDAO input/output indices for multiple bodies,
    with a separate MELD instance for each

    Parameters
    ----------
    comm : :class:`~mpi4py.MPI.Comm`
        The communicator object created for this xfer object instance
    isym : int
        Whether to search for symmetries in the geometry for transfer
    n : int
        The number of nearest neighbors included in the transfers
    beta : float
        The exponential decay factor used to average loads, displacements, etc.
    aero_node_ids : list
        List of aerodynamic node IDs on the current body and rank
    aero_nnodes : int
        Total number of aerodynamic nodes on the current rank
    struct_node_ids : list
        List of structural node IDs on the current body and rank
    struct_nnodes : int
        Total number of structural nodes on the current rank
    struct_ndof : int
        Number of degrees of freedom at each structural node
    linearized : bool
        Whether to use linearized MELD
    """

    def __init__(
        self,
        comm,
        isym=-1,
        n=200,
        beta=0.5,
        aero_node_ids=None,
        aero_nnodes=None,
        struct_node_ids=None,
        struct_nnodes=None,
        struct_ndof=None,
        linearized=False,
    ):
        # determine input/output indices for the current body
        if (
            struct_node_ids is not None
        ):  # use grid ids from struct_builder's get_tagged_indices
            # account for xyz of each node
            self.struct_coord_indices = np.zeros(len(struct_node_ids) * 3, dtype=int)
            for ii in range(3):
                self.struct_coord_indices[ii::3] = 3 * struct_node_ids + ii
            self.struct_coord_indices = list(self.struct_coord_indices)

            # account for structural DOFs of each node
            self.struct_dof_indices = np.zeros(
                len(struct_node_ids) * struct_ndof, dtype=int
            )
            for ii in range(struct_ndof):
                self.struct_dof_indices[ii::struct_ndof] = (
                    struct_node_ids * struct_ndof + ii
                )
            self.struct_dof_indices = list(self.struct_dof_indices)

        else:  # use all indices
            self.struct_coord_indices = list(np.arange(struct_nnodes * 3))
            self.struct_dof_indices = list(np.arange(struct_nnodes * struct_ndof))

        if (
            aero_node_ids is not None
        ):  # use grid ids from aero_builder's get_tagged_indices
            # account for xyz of each node
            self.aero_coord_indices = np.zeros(len(aero_node_ids) * 3, dtype=int)
            for ii in range(3):
                self.aero_coord_indices[ii::3] = 3 * aero_node_ids + ii
            self.aero_coord_indices = list(self.aero_coord_indices)

        else:  # use all indices
            self.aero_coord_indices = list(np.arange(aero_nnodes * 3))

        # new MELD instance
        if linearized:
            self.meld = TransferScheme.pyLinearizedMELD(
                comm, comm, 0, comm, 0, isym, n, beta
            )
        else:
            self.meld = TransferScheme.pyMELD(comm, comm, 0, comm, 0, isym, n, beta)
        self.initialized_meld = False


class MeldBuilder(Builder):
    def __init__(
        self,
        aero_builder,
        struct_builder,
        isym=-1,
        n=200,
        beta=0.5,
        check_partials=False,
        linearized=False,
        body_tags=None,
    ):
        self.aero_builder = aero_builder
        self.struct_builder = struct_builder
        self.isym = isym
        self.n = n
        self.beta = beta
        self.under_check_partials = check_partials
        self.linearized = linearized
        self.body_tags = body_tags if body_tags is not None else []

        if len(self.body_tags) > 0:  # make into lists, potentially for different bodies
            if not hasattr(self.n, "__len__"):
                self.n = [self.n] * len(self.body_tags)
            if not hasattr(self.beta, "__len__"):
                self.beta = [self.beta] * len(self.body_tags)
            if not hasattr(self.linearized, "__len__"):
                self.linearized = [self.linearized] * len(self.body_tags)

    def initialize(self, comm):
        self.nnodes_aero = self.aero_builder.get_number_of_nodes()
        self.nnodes_struct = self.struct_builder.get_number_of_nodes()
        self.ndof_struct = self.struct_builder.get_ndof()

        self.bodies = []

        if len(self.body_tags) > 0:  # body tags given
            for i in range(len(self.body_tags)):
                try:
                    aero_node_ids = np.atleast_1d(
                        self.aero_builder.get_tagged_indices(self.body_tags[i]["aero"])
                    )
                except NotImplementedError:
                    if comm.rank == 0:
                        print(
                            "get_tagged_indices has not been implemented in the aero builder; all nodes will be used"
                        )
                    aero_node_ids = None
                try:
                    struct_node_ids = np.atleast_1d(
                        self.struct_builder.get_tagged_indices(
                            self.body_tags[i]["struct"]
                        )
                    )
                except NotImplementedError:
                    if comm.rank == 0:
                        print(
                            "get_tagged_indices has not been implemented in the struct builder; all nodes will be used"
                        )
                    struct_node_ids = None
                self.bodies += [
                    MeldBodyInstance(
                        comm,
                        isym=self.isym,
                        n=self.n[i],
                        beta=self.beta[i],
                        aero_node_ids=aero_node_ids,
                        aero_nnodes=self.nnodes_aero,
                        struct_node_ids=struct_node_ids,
                        struct_nnodes=self.nnodes_struct,
                        struct_ndof=self.ndof_struct,
                        linearized=self.linearized[i],
                    )
                ]

        else:  # default: couple all nodes with a single MELD instance
            self.bodies = [
                MeldBodyInstance(
                    comm,
                    isym=self.isym,
                    n=self.n,
                    beta=self.beta,
                    aero_nnodes=self.nnodes_aero,
                    struct_nnodes=self.nnodes_struct,
                    struct_ndof=self.ndof_struct,
                    linearized=self.linearized,
                )
            ]

    def get_coupling_group_subsystem(self, scenario_name=None):
        disp_xfer = MeldDispXfer(
            struct_ndof=self.ndof_struct,
            struct_nnodes=self.nnodes_struct,
            aero_nnodes=self.nnodes_aero,
            check_partials=self.under_check_partials,
            bodies=self.bodies,
        )

        load_xfer = MeldLoadXfer(
            struct_ndof=self.ndof_struct,
            struct_nnodes=self.nnodes_struct,
            aero_nnodes=self.nnodes_aero,
            check_partials=self.under_check_partials,
            bodies=self.bodies,
        )

        return disp_xfer, load_xfer
