__all__ = ["HandcraftedMeshMorph"]

import numpy as np
from ...model._base import Base
from mpi4py import MPI
from funtofem import TransferScheme
from ...driver.transfer_settings import TransferSettings

class HandcraftedMeshMorph:
    def __init__(self, transfer_settings):
        self.transfer_settings = transfer_settings

    def initialize_transfer(
        self,
        comm,
        struct_comm,
        struct_root,
        aero_comm,
        aero_root,
        transfer_settings=None,
    ):
        """
        Initialize the load and displacement and/or thermal transfer scheme for this body

        Parameters
        ----------
        comm: MPI.comm
            MPI communicator
        transfer_settings: TransferSettings
            options for the load and displacement transfer scheme for the bodies
        """

        # If the user did not specify a transfer scheme default to MELD
        if transfer_settings is None:
            transfer_settings = TransferSettings()

        # Initialize the transfer and thermal transfer objects to None
        self.transfer = None
        self.thermal_transfer = None

        # Verify analysis type is valid
        self.verify_analysis_type(self.analysis_type)

        elastic_analyses = [_ for _ in Body.ANALYSIS_TYPES if "elastic" in _]
        thermal_analyses = [_ for _ in Body.ANALYSIS_TYPES if "therm" in _]

        # Set up the transfer schemes based on the type of analysis set for this body
        if self.analysis_type in elastic_analyses:
            # Set up the load and displacement transfer schemes
            if transfer_settings.elastic_scheme == "hermes":
                self.transfer = HermesTransfer(
                    self.comm, self.struct_comm, self.aero_comm
                )

            elif transfer_settings.elastic_scheme == "rbf":
                basis = TransferScheme.PY_THIN_PLATE_SPLINE

                if "basis function" in transfer_settings.options:
                    if (
                        transfer_settings.options["basis function"].lower()
                        == "thin plate spline"
                    ):
                        basis = TransferScheme.PY_THIN_PLATE_SPLINE
                    elif (
                        transfer_settings.options["basis function"].lower()
                        == "gaussian"
                    ):
                        basis = TransferScheme.PY_GAUSSIAN
                    elif (
                        transfer_settings.options["basis function"].lower()
                        == "multiquadric"
                    ):
                        basis = TransferScheme.PY_MULTIQUADRIC
                    elif (
                        transfer_settings.options["basis function"].lower()
                        == "inverse multiquadric"
                    ):
                        basis = TransferScheme.PY_INVERSE_MULTIQUADRIC
                    else:
                        print("Unknown RBF basis function for body number")
                        quit()

                self.transfer = TransferScheme.pyRBF(
                    comm, struct_comm, struct_root, aero_comm, aero_root, basis, 1
                )

            elif transfer_settings.elastic_scheme == "meld":
                self.transfer = TransferScheme.pyMELD(
                    comm,
                    struct_comm,
                    struct_root,
                    aero_comm,
                    aero_root,
                    transfer_settings.isym,
                    transfer_settings.npts,
                    transfer_settings.beta,
                )

            elif transfer_settings.elastic_scheme == "linearized meld":
                self.transfer = TransferScheme.pyLinearizedMELD(
                    comm,
                    struct_comm,
                    struct_root,
                    aero_comm,
                    aero_root,
                    transfer_settings.isym,
                    transfer_settings.npts,
                    transfer_settings.beta,
                )

            elif transfer_settings.elastic_scheme == "beam":
                self.xfer_ndof = transfer_settings.options["ndof"]
                self.transfer = TransferScheme.pyBeamTransfer(
                    comm,
                    struct_comm,
                    struct_root,
                    aero_comm,
                    aero_root,
                    transfer_settings.options["conn"],
                    transfer_settings.options["nelems"],
                    transfer_settings.options["order"],
                    transfer_settings.options["ndof"],
                )
            else:
                print("Error: Unknown transfer scheme for body")
                quit()

        # Set up the transfer schemes based on the type of analysis set for this body
        if self.analysis_type in thermal_analyses:
            # Set up the thermal transfer schemes

            if transfer_settings.thermal_scheme == "meld":
                self.thermal_transfer = TransferScheme.pyMELDThermal(
                    comm,
                    struct_comm,
                    struct_root,
                    aero_comm,
                    aero_root,
                    transfer_settings.isym,
                    transfer_settings.thermal_npts,
                    transfer_settings.thermal_beta,
                )
            else:
                print("Error: Unknown thermal transfer scheme for body")
                quit()

        # Set the node locations
        self.update_transfer()

        # Initialize the load/displacement transfer
        if self.transfer is not None:
            self.transfer.initialize()

        # Initialize the thermal transfer
        if self.thermal_transfer is not None:
            self.thermal_transfer.initialize()

        return