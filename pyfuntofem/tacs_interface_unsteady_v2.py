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

from __future__ import print_function
from tkinter.tix import INTEGER

from tacs             import TACS, functions
from tacs_builder     import TACSBodyType
from .solver_interface import SolverInterface
from typing import TYPE_CHECKING

import numpy as np

class IntegrationSettings:
    INTEGRATION_TYPES = ['BDF','DIRK']

    def __init__(self,
        integration_type:str = 'BDF',
        integration_order:int=2,
        L2_convergence:float=1e-12,
        L2_convergence_rel:float=1e-12,
        jac_assembly_freq:int=1,
        write_solution:bool=True,
        number_solution_files:bool=True,
        print_timing_info:bool=False,
        print_level:int = 0
    ):
        # TODO : add comments for this 
        """
        
        """
        assert(integration_type in IntegrationSettings.INTEGRATION_TYPES)
        
        self.integration_type = integration_type
        self.integration_order = integration_order
        self.L2_convergence = L2_convergence
        self.L2_convergence_rel = L2_convergence_rel
        self.jac_assembly_freq = jac_assembly_freq
        self.write_solution = write_solution
        self.number_solution_files = number_solution_files
        self.print_timing_info = print_timing_info
        self.print_level = print_level

    @property
    def is_bdf(self) -> bool:
        return self.integration_type == 'BDF'

    @property
    def is_dirk(self) -> bool:
        return self.integration_type == 'DIRK'

    @property
    def num_stages(self) -> int:
        return self.integration_order - 1

class TacsOutputGeneratorUnsteady:
    def __init__(self, path, name="tacs_output", f5=None):
        self.path = path
        self.name = name
        self.f5 = f5
        # TODO : complete this class

    def __call__(self):
        # TODO : write f5 files for each time step
        pass


class TacsUnsteadyInterface(SolverInterface):
    """
    A base class to do coupled unsteady simulations with TACS
    """

    def __init__(self, 
                 comm,
                 model,
                 assembler=None,
                 gen_output:TacsOutputGeneratorUnsteady=None,
                 thermal_index:int=0,
                 struct_id:int=None,
                 integration_settings:IntegrationSettings=None
    ):

        self.comm = comm
        self.tacs_comm = None

        # get active design variables
        self.variables = model.get_variables()
        self.struct_variables = []
        for var in self.variables:
            if var.analysis_type == "structural":
                self.struct_variables.append(var)

        self.integration_settings = integration_settings
        self.gen_output = gen_output

        # initialize variables
        self._initialize_variables(
            model, assembler, thermal_index=thermal_index, struct_id=struct_id
        )

    def _initialize_variables(
        self,
        model,
        assembler=None,
        mat=None,
        pc=None,
        gmres=None,
        struct_id=None,
        thermal_index=0,
    ):

        self.thermal_index = thermal_index
        self.struct_id = struct_id

        # Boolean indicating whether TACSAssembler is on this processor
        # or not. If not, all variables are None.
        self.tacs_proc = False

        # Assembler object
        self.assembler = None

        # TACS vectors
        self.res = None
        self.ans = None
        self.ext_force = None
        self.update = None

        # Matrix, preconditioner and solver method
        self.mat = None
        self.pc = None
        self.gmres = None

        # setup the integrator looping over each of the scenarios
        self.integrator = {}
        for scenario in model.scenarios:
            self.integrator[scenario.id] = self.create

            # Create the time integrator and allocate the load data structures
            if self.integration_settings.is_bdf:
                self.integrator = TACS.BDFIntegrator(self.assembler, self.tInit, self.tFinal,
                                                        float(self.numSteps), self.integration_settings.integration_order)
                # Create a force vector for each time step
                self.F = [self.assembler.createVec() for i in range(self.numSteps + 1)]
                # Auxillary element object for applying tractions/pressure
                self.auxElems = [TACS.AuxElements() for i in range(self.numSteps + 1)]

            elif self.integration_settings.is_dirk:
                self.numStages = self.integration_settings.num_stages
                self.integrator = TACS.DIRKIntegrator(self.assembler, self.tInit, self.tFinal,
                                                        float(self.numSteps), self.numStages)
                # Create a force vector for each time stage
                self.F = [self.assembler.createVec() for i in range((self.numSteps + 1)*self.numStages)]
                # Auxiliary element object for applying tractions/pressure at each time stage
                self.auxElems = [TACS.AuxElements() for i in range((self.numSteps + 1)*self.numStages)]

            


def createTacsUnsteadyInterfaceFromBDF(
    model,
    comm,
    nprocs,
    bdf_file,
    t0=0.0,
    tf=1.0,
    prefix="",
    callback=None,
    struct_options={},
    thermal_index=-1,
):
    # TODO : determine if inputs should be t0,tf or nsteps, dt
    """
    Create a TacsSteadyInterface instance using the pytacs BDF loader

    Parameters
    ----------
    model: :class:`FUNtoFEMmodel`
        The model class associated with the problem
    comm: MPI.comm
        MPI communicator (typically MPI_COMM_WORLD)
    bdf_file: str
        The BDF file name
    prefix: 

    callback: function
        The element callback function for pyTACS
    struct_options: dictionary
        The options passed to pyTACS
    """

    # Split the communicator
    world_rank = comm.Get_rank()
    if world_rank < nprocs:
        color = 1
    else:
        color = MPI.UNDEFINED
    tacs_comm = comm.Split(color, world_rank)

    assembler = None
    f5 = None
    if world_rank < nprocs:
        # Create the assembler class
        fea_assembler = pytacs.pyTACS(bdf_file, tacs_comm, options=struct_options)

        # Set up constitutive objects and elements
        fea_assembler.initialize(callback)

        # Set the assembler
        assembler = fea_assembler.assembler

        # Set the output file creator
        f5 = fea_assembler.outputViewer

    # Create the output generator
    gen_output = TacsOutputGenerator(prefix, f5=f5)

    # We might need to clean up this code. This is making educated guesses
    # about what index the temperature is stored. This could be wrong if things
    # change later. May query from TACS directly?
    if assembler is not None and thermal_index == -1:
        varsPerNode = assembler.getVarsPerNode()

        # This is the likely index of the temperature variable
        if varsPerNode == 1:  # Thermal only
            thermal_index = 0
        elif varsPerNode == 4:  # Solid + thermal
            thermal_index = 3
        elif varsPerNode >= 7:  # Shell or beam + thermal
            thermal_index = 3

    # Broad cast the thermal index to ensure it's the same on all procs
    thermal_index = comm.bcast(thermal_index, root=0)

    # Create the tacs interface
    interface = TacsUnsteadyInterface(
        comm, model, assembler, gen_output, thermal_index=thermal_index
    )

    return interface

        
