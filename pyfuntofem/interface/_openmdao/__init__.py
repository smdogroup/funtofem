# __init__ for pyfuntofem/openmdao subfolder

# checks whether openmdao is available
import importlib
openmdao_loader = importlib.util.find_spec('openmdao')
has_openmdao = openmdao_loader is not None

# if openmdao module is available, proceed
if has_openmdao:
    from .openmdao_component import *
