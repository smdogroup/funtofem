#!/usr/bin/env python
"""
This file is part of the package FUNtoFEM for coupled aeroelastic simulation
and design optimization.

Copyright (C) 2015 Georgia Tech Research Corporation.
Additional copyright (C) 2015 Kevin Jacobson, Jan Kiviaho and Graeme Kennedy.
All rights reserved.

FUNtoFEM is licensed under the Apache License, Version 2.0 (the "License");
you may not use this software except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# import each analysis interface subfolder
# each analysis subfolder checks whether the package is available before import

# use importlib to check available packages for interface imports
import importlib

# import the base interface and solver manager
from ._solver_interface import *
from .solver_manager import *

# Import all of the funtofem analysis interfaces
# ----------------------------------------------

# Cart3D interface
from .cart3d_interface import *

# caps2fun utils
from .caps2fun import *

# Radiation interface
from .radiation_interface import *

# FUN3D interface, with python package "fun3d"
fun3d_loader = importlib.util.find_spec("fun3d")
if fun3d_loader is not None:
    from .fun3d_interface import *
    from .fun3d_14_interface import *

# SU2 interface, with python package "pysu2"
su2_loader = importlib.util.find_spec("pysu2")
if su2_loader is not None:
    from .su2_interface import *

# TACS interface, with python packaage "tacs"
tacs_loader = importlib.util.find_spec("tacs")
if tacs_loader is not None:
    from .tacs_interface import *
    from .tacs_interface_unsteady import *

# test interfaces
from .test_interfaces import *

# import any interface utilities
# -------------------------------
from .utils import *
