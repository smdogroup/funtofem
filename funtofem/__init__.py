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

# import each subfolder
# each analysis subfolder checks whether the package is available before import

# TIP : open a python shell and run the following to check imported packages:
# import funtofem
# funtofem.__dict__

import importlib

from .driver import *
from .interface import *
from .model import *
from .optimization import *

mphys_loader = importlib.util.find_spec("mphys")
openmdao_loader = importlib.util.find_spec("openmdao")
if mphys_loader is not None and openmdao_loader is not None:
    try:  # sometimes openmdao import fails on unittests
        from .mphys import *
    except:
        print(
            "Mphys module couldn't be built despite available openmdao and mphys packages."
        )
