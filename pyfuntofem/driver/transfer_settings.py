
__all__ = ["TransferSettings"]

from typing import TYPE_CHECKING

class TransferSettings:
    ELASTIC_SCHEMES = ["hermes", "rbf", "meld", "linearized meld", "beam"]
    THERMAL_SCHEMES = ["meld"]
    # TODO : determine whether we should put analysis type back in transfer settings?
    def __init__(
        self, 
        elastic_scheme="meld", 
        thermal_scheme="meld", 
        npts:int=200, 
        beta:float=0.5, 
        isym:int=-1,
        options:dict={},
    ):
        """
        Transfer settings for the fully-coupled funtofem analysis across different fluid, structural meshes
        Parameters
        ---------------------------------        
        elastic_scheme: the transfer scheme used for loads, displacements
        thermal_scheme: the transfer scheme used for heat loads, temperatures
        npts: the number of nearest neighbors included in the transfers
        beta: the exponential decay factor used to average loads, displacements, etc.
        isym: whether to search for symmetries in the geometry for transfer
        options: additional options dictionary like for beam and rbf basis functions
        """
        assert(elastic_scheme in ELASTIC_SCHEMES)
        assert(thermal_scheme in THERMAL_SCHEMES)
        self.elastic_scheme = elastic_scheme
        self.thermal_scheme = thermal_scheme
        self.npts = npts
        self.beta = beta
        self.isym = isym

    def scheme(self, new_scheme):
        """
        elastic and thermal scheme setter with method cascading
        """
        self.elastic_scheme = new_scheme
        self.thermal_scheme = new_scheme
        return self

    def elastic_scheme(self, new_scheme):
        """
        elastic scheme setter with method cascading
        """
        self.elastic_scheme = new_scheme
        return self

    def thermal_scheme(self, new_scheme):
        """
        thermal scheme setter with method cascading
        """
        self.thermal_scheme = new_scheme
        return self
