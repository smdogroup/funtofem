__all__ = ["TransferSettings"]

from typing import TYPE_CHECKING, ClassVar
from dataclasses import dataclass
from funtofem import TransferScheme


@dataclass  # dataclass decorator includes any variables below as fields and in the constructor
class TransferSettings:
    # class attributes, not included in the constructor
    ELASTIC_SCHEMES: ClassVar[str] = [
        "hermes",
        "rbf", 
        "meld",
        "linearized meld",
        "beam",
    ]
    THERMAL_SCHEMES: ClassVar[str] = ["meld"]

    # arguments put in the constructor with @dataclass
    elastic_scheme: str = "meld"  # elastic scheme for loads, displacements
    thermal_scheme: str = "meld"  # transfer scheme for heat loads, temperatures
    npts: int = 200  # number of nearest neighbors
    beta: float = 0.5  # exp decay factor of neighbors
    isym: int = -1  # flag to use symmetries in transfer

    def __post__init__(self):
        # check if inputs are valid
        assert(self.elastic_scheme in TransferSettings.ELASTIC_SCHEMES)
        assert(self.thermal_scheme in TransferSettings.THERMAL_SCHEMES)
        return

@dataclass
class RbfTransferSettings(TransferSettings):
    # RBF Class options
    BASIS_FUNCTIONS: ClassVar[object] = [TransferScheme.PY_THIN_PLATE_SPLINE, TransferScheme.PY_GAUSSIAN, TransferScheme.PY_MULTIQUADRIC, TransferScheme.PY_INVERSE_MULTIQUADRIC]

    # Additional constructor arguments
    basis_function = TransferScheme.PY_THIN_PLATE_SPLINE

@dataclass
class BeamTransferSettings(TransferSettings):
    # Beam Transfer Settings
    conn = []
    nelems = 10
    order = 1
    ndof = 10