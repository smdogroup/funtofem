# Subfolder for pyfuntofem/_su2 interface
# classes / methods in include * are found in __all__ header

# check whether su2 exists
import importlib
su2_loader = importlib.util.find_spec('pysu2')
has_su2 = su2_loader is not None

# if has su2, proceed with imports
if has_su2:
    from .su2_interface import *
