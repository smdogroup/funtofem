
# import and augment caps2tacs module from TACS repo for funtofem
from tacs.caps2tacs import *
from pyfuntofem.model.funtofem_model import FUNtoFEMmodel

# add register to method to tacs model with funtofem model
def register_to(self, obj):
    if isinstance(obj, FUNtoFEMmodel):
        obj.tacs_model = self
    else:
        raise AssertionError("Can't register TacsModel to an object other than FUNtoFEMmodel")
TacsModel.register_to = register_to

