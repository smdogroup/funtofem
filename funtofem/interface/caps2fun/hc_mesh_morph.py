__all__ = ["HandcraftedMeshMorph"]

import numpy as np
from funtofem import TransferScheme
from mpi4py import MPI
import os


class HandcraftedMeshMorph:
    def __init__(self, comm, model, transfer_settings, nprocs_hc=1, auto_coords=True):
        self.comm = comm
        self.model = model
        self.nprocs_hc = nprocs_hc
        self.transfer_settings = transfer_settings
        self.auto_coords = auto_coords

        # initialize handcrafted aero surf mesh coords
        self.hc_aero_X = None
        self.hc_aero_id = None
        self.hc_nnodes = 0
        self.first_body = self.model.bodies[0]
        if auto_coords:
            self._get_hc_coords()

        self._first_caps_read = True
        self.caps_aero_X = None
        self.caps_aero_id = None
        self.caps_nnodes = 0

        # TODO : can maybe extend this later to use more than this many procs, but not yet
        assert nprocs_hc == 1

        self.u_caps = None  # caps shape change displacement
        self.u_hc = None  # handcrafted mesh shape change displacement

        self.caps_aero_shape_term = {}
        for scenario in self.model.scenarios:
            self.caps_aero_shape_term[scenario.id] = None

        self._flow_dir = None

    @property
    def flow_dir(self):
        return self._flow_dir

    @flow_dir.setter
    def flow_dir(self, new_dir):
        self._flow_dir = new_dir

    @property
    def hc_mesh_morph_filepath(self):
        return os.path.join(self.flow_dir, "hc_surface.dat")

    def _get_hc_coords(self):
        """get the handcrafted aero surface mesh coords and ids"""

        # just use the first body for now (not fully general but that's ok)
        first_body = self.first_body

        if first_body.aero_X is None or first_body.aero_id is None:
            print(
                "Funtofem warning : need to build handcrafted mesh morph file after Fun3dInterface which reads aero surf coordinates."
            )
            return

        self.hc_aero_X = first_body.aero_X
        self.hc_aero_id = first_body.aero_id
        self.hc_nnodes = self.hc_aero_X.shape[0] // 3  # // produces an int
        return

    def read_surface_file(self, surface_morph_file, is_caps_mesh=True):
        # read the file here
        aero_X = None
        aero_id = None
        if self.comm.rank == 0:
            fp = open(surface_morph_file, "r")
            lines = fp.readlines()
            fp.close()

            aero_X = []
            aero_id = []
            nnodes = None
            inode = None
            # TODO : do we need to read the element connectivity => probably not

            for line in lines:
                if inode is not None and inode < nnodes:
                    inode += 1
                    chunks = line.split(" ")
                    # add the xyz coords and aero id
                    aero_X += [float(chunks[0]), float(chunks[1]), float(chunks[2])]
                    aero_id += [int(chunks[3])]

                if "title" in line or "variables" in line:
                    continue
                elif "zone" in line:
                    chunks = line.split(" ")
                    for chunk in chunks:
                        if "i=" in chunk:
                            nnodes = int(chunk.split("=")[1].split(",")[0])
                    inode = 0

            # convert to numpy arrays
            aero_X = np.array(aero_X, dtype=TransferScheme.dtype)
            aero_id = np.array(aero_id, dtype=int)
        else:
            aero_X = np.zeros((0,), dtype=TransferScheme.dtype)
            aero_id = np.zeros((0,), dtype=int)

        # TODO : distribute these nodes and ids among the nprocs_hc next (if using more than one proc for this)

        self.comm.Barrier()

        if is_caps_mesh:
            if self._first_caps_read:
                self._first_caps_read = False

                # copy the aero_X, aero_id
                # TODO : distribute these arrays across multiple procs if need be
                if self.comm.rank == 0:
                    self.caps_aero_X = aero_X
                    self.caps_aero_id = aero_id
                    self.caps_nnodes = aero_X.shape[0] // 3
                else:
                    self.caps_nnodes = 0
                    self.caps_aero_X = np.zeros((0,), dtype=TransferScheme.dtype)
                    self.caps_aero_id = np.zeros((self.caps_nnodes,), dtype=int)

                # initialize the transfer object
                self.transfer = None
                self._initialize_transfer()

                # initial zero displacements
                self.u_caps = self.caps_aero_X * 0.0
            else:
                if self.comm.rank == 0:
                    nnodes = aero_X.shape[0] // 3
                    assert nnodes == self.caps_nnodes

                # otherwise save shape changing displacements
                self.u_caps = aero_X - self.caps_aero_X
        else:  # reading an hc mesh with the surface dat file, this feature mostly just for testing
            # since it can deform the farfield that we don't want to deform..

            # copy the aero_X, aero_id
            if self.comm.rank == 0:
                self.hc_aero_X = aero_X
                self.hc_aero_id = aero_id
                self.hc_nnodes = aero_X.shape[0] // 3
            else:
                self.hc_nnodes = 0
                self.hc_aero_X = np.zeros((0,), dtype=TransferScheme.dtype)
                self.hc_aero_id = np.zeros((self.hc_nnodes,), dtype=int)

        return

    def _distribute_hc_test_mesh(self, root=0):
        """distribute a handcrafted test mesh (only for unittesting) across all processors like FUN3D is"""
        size = self.comm.Get_size()
        if self.comm.rank == root:
            aero_X = self.hc_aero_X
            aero_id = self.hc_aero_id

            print(f"orig hc nnodes = {self.hc_nnodes}")

            X_list = []
            id_list = []
            for irank in range(size):
                xvals = aero_X[0::3]
                yvals = aero_X[1::3]
                zvals = aero_X[2::3]
                loc_xvals = xvals[irank::size]
                loc_yvals = yvals[irank::size]
                loc_zvals = zvals[irank::size]
                loc_aero_X = []
                for i in range(loc_xvals.shape[0]):
                    loc_aero_X += [loc_xvals[i], loc_yvals[i], loc_zvals[i]]
                loc_aero_X = np.array(loc_aero_X, dtype=TransferScheme.dtype)
                X_list += [loc_aero_X]
                id_list += [aero_id[irank::size]]

        else:
            X_list = None
            id_list = None

        self.hc_aero_X = self.comm.scatter(X_list, root=root)
        self.hc_aero_id = self.comm.scatter(id_list, root=root)
        self.hc_nnodes = self.hc_aero_id.shape[0]

        print(f"rank {self.comm.rank} nnodes = {self.hc_nnodes}")
        return

    def _initialize_transfer(self):
        """
        Initialize the load and displacement and/or thermal transfer scheme for this body

        Parameters
        ----------
        comm: MPI.comm
            MPI communicator
        transfer_settings: TransferSettings
            options for the load and displacement transfer scheme for the bodies
        """

        # make a comm for the handcrafted mesh limited to how many procs it uses (default is just 1)
        world_rank = self.comm.Get_rank()
        if world_rank < self.nprocs_hc:
            color = 1
        else:
            color = MPI.UNDEFINED
        caps_comm = self.comm.Split(color, world_rank)
        # self.hc_comm = hc_comm # both on one proc comms for meld

        # Initialize the transfer transfer objects
        self.transfer = TransferScheme.pyMELD(
            self.comm,
            caps_comm,  # caps comm limited to root proc
            0,
            self.comm,  # use regular comm for aero handcrafted mesh
            0,
            self.transfer_settings.isym,
            self.transfer_settings.npts,
            self.transfer_settings.beta,
        )

        if self.comm.rank == 0:
            assert self.hc_aero_X is not None
            assert self.caps_aero_X is not None

        # Set the node locations
        # CAPS mesh treated as structure mesh and HC mesh as aero
        self.transfer.setStructNodes(self.caps_aero_X)
        self.transfer.setAeroNodes(self.hc_aero_X)

        self.transfer.initialize()

        return

    def transfer_shape_disps(self):
        """transfer the shape changing displacements from the CAPS to the handcrafted mesh"""
        # reset hc aero displacements
        self.u_hc = np.zeros((3 * self.hc_nnodes), dtype=TransferScheme.dtype)

        self.transfer.transferDisps(self.u_caps, self.u_hc)

        print(f"rank {self.comm.rank} u_hc = {self.u_hc} avg {np.mean(self.u_hc)}")

        # also transfer the loads since adjoint sensitivities require this (virtual work computation)
        # but just transfer zero loads since we only care about disp transfer here
        hc_loads = 0.0 * self.u_hc
        caps_loads = np.zeros((3 * self.caps_nnodes), dtype=TransferScheme.dtype)
        self.transfer.transferLoads(hc_loads, caps_loads)

        return

    @property
    def hc_def_aero_X(self):
        return self.hc_aero_X + self.u_hc

    def write_surface_file(self, surface_morph_file=None):
        """write a surface mesh morphing file for the Handcrafted mesh"""
        if surface_morph_file is None:  # use default filepath if not specified
            surface_morph_file = self.hc_mesh_morph_filepath

        # need to collect aero coordinates and ids from all nodes
        hc_aero_X, hc_aero_ids = self._collect_hc_deformed_coords(root=0)

        if self.comm.rank == 0:
            fp = open(surface_morph_file, "w")

            hc_nnodes = hc_aero_ids.shape[0]

            # first write the headers
            fp.write("title=" "CAPS" "\n")
            fp.write("variables=" "x" "," "y" "," "z" "," "id" "\n")
            fp.write(
                "zone t="
                "Body_1"
                ", i="
                "" + f"{hc_nnodes}"
                ", j=0, f=fepoint, solutiontime=0.000000, strandid=0\n"
            )

            # then write each of the nodes
            for i in range(hc_nnodes):
                xyz = np.real(hc_aero_X[3 * i : 3 * i + 3])
                nid = np.real(hc_aero_ids[i])
                fp.write(f"{xyz[0]:3.16e} {xyz[1]:3.16e} {xyz[2]:3.16e} {nid}\n")

            fp.close()

        self.comm.Barrier()
        return

    def compute_caps_coord_derivatives(self, scenario):
        """transfer the aero coordinate derivatives from the handcrafted mesh back to the native CAPS mesh"""
        nfunctions = scenario.count_adjoint_functions()
        aero_shape_term = self.first_body.get_aero_coordinate_derivatives(scenario)
        self.caps_aero_shape_term[scenario.id] = np.zeros(
            (3 * self.caps_nnodes, nfunctions), dtype=TransferScheme.dtype
        )
        caps_aero_shape_term = self.caps_aero_shape_term[scenario.id]
        temp_xcaps = np.zeros((3 * self.caps_nnodes), dtype=TransferScheme.dtype)

        for k in range(nfunctions):
            # self.transfer.applydDdxS0(1.0 * aero_shape_term[:, k], temp_xcaps)
            self.transfer.applydDduSTrans(1.0 * aero_shape_term[:, k], temp_xcaps)
            caps_aero_shape_term[:, k] -= temp_xcaps
            # caps_aero_shape_term[:, k] += temp_xcaps

    def _collect_caps_coordinate_derivatives(self, root=0):
        all_aero_ids = self.comm.gather(self.caps_aero_id, root=0)

        # append struct shapes for each scenario
        full_aero_shape_term = []
        for scenario in self.model.scenarios:
            full_aero_shape_term.append(self.caps_aero_shape_term[scenario.id])
        full_aero_shape_term = np.concatenate(full_aero_shape_term, axis=1)

        all_aero_shape = self.comm.gather(full_aero_shape_term, root=root)

        aero_ids = []
        aero_shape = []

        if self.comm.rank == root:
            # Discard any entries that are None
            aero_ids = []
            for d in all_aero_ids:
                if d is not None:
                    aero_ids.append(d)

            aero_shape = []
            for d in all_aero_shape:
                if d is not None:
                    aero_shape.append(d)

            if len(aero_shape) > 0:
                aero_shape = np.concatenate(aero_shape)
            else:
                aero_shape = np.zeros((3, 1))

            if len(aero_ids) == 0:
                aero_ids = np.arange(aero_shape.shape[0] // 3, dtype=int)
            else:
                aero_ids = np.concatenate(aero_ids)

        return aero_shape, aero_ids

    def _collect_hc_deformed_coords(self, root=0):
        all_aero_ids = self.comm.gather(self.hc_aero_id, root=0)
        all_aero_X = self.comm.gather(self.hc_def_aero_X, root=0)

        aero_ids = []
        aero_X = []

        if self.comm.rank == root:
            # Discard any entries that are None
            aero_ids = []
            for d in all_aero_ids:
                if d is not None:
                    aero_ids.append(d)

            aero_X = []
            for d in all_aero_X:
                if d is not None:
                    aero_X.append(d)

            if len(aero_X) > 0:
                aero_X = np.concatenate(aero_X)
            else:
                aero_X = np.zeros((3, 1))

            if len(aero_ids) == 0:
                aero_ids = np.arange(aero_X.shape[0] // 3, dtype=int)
            else:
                aero_ids = np.concatenate(aero_ids)

        return aero_X, aero_ids

    def write_sensitivity_file(
        self, comm, filename, discipline="aerodynamic", root=0, write_dvs: bool = True
    ):
        """
        Write the sensitivity file.

        This file contains the following information:

        Number of functionals

        Functional name
        Number of surface nodes
        for node in surface_nodes:
            node, dfdx, dfdy, dfdz

        Parameters
        ----------
        comm: MPI communicator
            Global communicator across all FUNtoFEM processors
        filename: str
            The name of the file to be generated
        discipline: str
            The name of the discipline sensitivity data to be written
        root: int
            The rank of the processor that will write the file
        write_dvs: bool
            whether to write the design variables for this discipline
        """

        funcs = self.model.get_functions()

        deriv, id = self._collect_caps_coordinate_derivatives(root=root)

        if comm.rank == root:
            variables = self.model.get_variables()
            discpline_vars = []
            count = len(id)

            if write_dvs:  # flag for registering dvs that will later get written out
                for var in variables:
                    # Write the variables whose analysis_type matches the discipline string.
                    if discipline == var.analysis_type and var.active:
                        discpline_vars.append(var)

            # Write out the number of sets of discpline variables
            num_dvs = len(discpline_vars)

            # Write out the number of functionals and number of design variables
            data = "{} {}\n".format(len(funcs), num_dvs)

            for n, func in enumerate(funcs):
                # Print the function name
                data += "{}\n".format(func.full_name)

                # Print the function value
                data += "{}\n".format(func.value.real)

                # Print the number of coordinates
                data += "{}\n".format(count)

                for i in range(count):
                    data += "{} {} {} {}\n".format(
                        int(id[i]),
                        deriv[3 * i, n].real,
                        deriv[3 * i + 1, n].real,
                        deriv[3 * i + 2, n].real,
                    )

                for var in discpline_vars:
                    deriv = func.get_gradient_component(var)
                    deriv = deriv.real

                    # Write the variable name and derivative value
                    data += var.name + "\n"
                    data += "1\n"
                    data += str(deriv) + "\n"

            with open(filename, "w") as fp:
                fp.write(data)

        return
