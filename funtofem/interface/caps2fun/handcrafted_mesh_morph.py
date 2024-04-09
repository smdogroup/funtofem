__all__ = ["HandcraftedMeshMorph"]

import numpy as np
from funtofem import TransferScheme
from mpi4py import MPI

class HandcraftedMeshMorph:
    def __init__(self, comm, model, transfer_settings, nprocs_hc=1):
        self.comm = comm
        self.model = model
        self.nprocs_hc = nprocs_hc
        self.transfer_settings = transfer_settings

        # initialize handcrafted aero surf mesh coords
        self.hc_aero_X = None
        self.hc_aero_id = None
        self.hc_nnodes = 0
        self._get_hc_coords()

        self._first_caps_read = True
        self.caps_aero_X = None
        self.caps_aero_id = None
        self.caps_nnodes = 0

        self.u_caps = None # caps shape change displacement
        self.u_hc = None # handcrafted mesh shape change displacement

    def _get_hc_coords(self):
        """get the handcrafted aero surface mesh coords and ids"""

        # just use the first body for now (not fully general but that's ok)
        first_body = self.model.bodies[0]

        if first_body.aero_X is None or first_body.aero_id is None:
            print("Funtofem warning : need to build handcrafted mesh morph file after Fun3dInterface which reads aero surf coordinates.")
            return

        self.hc_aero_X = first_body.aero_X
        self.hc_aero_id = first_body.aero_id
        self.hc_nnodes = self.hc_aero_X.shape[0] // 3 # // produces an int
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
            aero_id = np.array(aero_id, dtype=TransferScheme.dtype)

        # TODO : distribute these nodes and ids among the nprocs_hc next (if using more than one proc for this)

        self.comm.Barrier()

        if is_caps_mesh:
            if self._first_caps_read:
                self._first_caps_read = False

                # copy the aero_X, aero_id
                self.caps_aero_X = aero_X
                self.caps_aero_id = aero_id
                self.caps_nnodes = aero_X.shape[0] // 3

                # initialize the transfer object
                self.transfer = None
                self._initialize_transfer()
            else:
                nnodes = aero_X.shape[0] // 3
                assert nnodes == self.caps_nnodes

                # otherwise save shape changing displacements
                self.u_caps = aero_X - self.caps_aero_X
        else: # reading an hc mesh with the surface dat file, this feature mostly just for testing
            # since it can deform the farfield that we don't want to deform..

            # copy the aero_X, aero_id
            self.hc_aero_X = aero_X
            self.hc_aero_id = aero_id
            self.hc_nnodes = aero_X.shape[0] // 3

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
        hc_comm = self.comm.Split(color, world_rank)

        # Initialize the transfer transfer objects
        self.transfer = TransferScheme.pyMELD(
            self.comm,
            hc_comm,
            0,
            self.comm,
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
        self.u_hc = np.zeros((3*self.hc_nnodes), dtype=TransferScheme.dtype)

        self.transfer.transferDisps(self.u_caps, self.u_hc)

        print(f"u caps = {self.u_caps}")

        # also transfer the loads since adjoint sensitivities require this (virtual work computation)
        # but just transfer zero loads since we only care about disp transfer here
        hc_loads = 0.0 * self.u_hc
        caps_loads = np.zeros((3*self.caps_nnodes), dtype=TransferScheme.dtype)
        self.transfer.transferLoads(hc_loads, caps_loads)

        return
    
    @property
    def hc_def_aero_X(self):
        return self.hc_aero_X + self.u_hc

    def write_surface_file(self, surface_morph_file):
        """write a surface mesh morphing file for the Handcrafted mesh"""
        if self.comm.rank == 0:
            fp = open(surface_morph_file, "w")
            
            # first write the headers
            fp.write("title=""CAPS""\n")
            fp.write("variables=""x"",""y"",""z"",""id""\n")
            fp.write("zone t=""Body_1"", i=""" + f"{self.hc_nnodes}"", j=0, f=fepoint, solutiontime=0.000000, strandid=0\n")

            hc_def_aero_X = self.hc_def_aero_X

            # then write each of the nodes
            for i in range(self.hc_nnodes):
                xyz = np.real(hc_def_aero_X[3*i:3*i+3])
                nid = np.real(self.hc_aero_id[i])
                fp.write(f"{xyz[0]:3.16e} {xyz[1]:3.16e} {xyz[2]:3.16e} {nid}\n")

            fp.close()

        self.comm.Barrier()
        return