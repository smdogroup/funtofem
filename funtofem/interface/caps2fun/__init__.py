# check the pyCAPS module can be loaded and only load remaining items if available
import importlib

caps_loader = importlib.util.find_spec("pyCAPS")

from .aflr_aim import *
from .fun3d_aim import *
from .hc_mesh_morph import *

if caps_loader is not None:
    from .fun3d_model import *
