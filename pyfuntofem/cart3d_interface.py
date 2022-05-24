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

import numpy as np
import os
from funtofem         import TransferScheme
from .solver_interface import SolverInterface
from .cart3d_utils     import ReadTriangulation, ComputeAeroLoads, WriteTri, RMS

class Cart3DInterface(SolverInterface):
    """
    FUNtoFEM interface class for Cart3D. Works for steady analysis only.
    Use of this interface requires that the user:
     - install Cart3D;
     - append the $Cart3D/bin $Cart3D/bin/$Cart3D_ARCH directories to $PATH
       variable
     - create a cart3d subdirectory in the run directory
     - add the appropriate input.c3d and input.cntl files to cart3d/
     - link the original triangulation as Components.i.tri in cart3d/
     - copy the aero.csh shell script into cart3d/

    """

    def __init__(self, comm, model, pinf, gamma, conv_hist=False, adapt_growth=None):
        """
        The instantiation of the Cart3D interface class will populate the model
        with the aerodynamic surface mesh, body.aero_X and body.aero_nnodes

        Parameters
        ----------
        comm: MPI.comm
            MPI communicator
        model : :class:`FUNtoFEMmodel`
            FUNtoFEM model
        comps : list
            list of components to use from Cart3D
        pinf : float
            freestream pressure
        gamma : float
            ratio of specific heats
        conv_hist : bool
            output convergence history to file
        adapt_growth : list
            list of number of adaptation cycles to do each aeroelastic iteration

        """
        self.comm = comm
        self.pinf = pinf
        self.gamma = gamma
        self.conv_hist = conv_hist
        self.adapt_growth = adapt_growth

        self.original_tri = None

        # Store previous iteration's aerodynamic forces and displacements for
        # each body in order to compute RMS error at each iteration
        if self.conv_hist:
            self.uprev = {}
            self.fprev = {}
            self.conv_hist_file = "conv_hist.dat"

        # Get the initial aerodynamic surface meshes
        self.initialize(model.scenarios[0], model.bodies)

    def initialize(self, scenario, bodies):
        """
        Runs aero.csh just to get the surface triangulation read
        into FUNtoFEM

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario that needs to be initialized
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies to either get new surface meshes from or to
            set the original mesh in

        Returns
        -------
        fail: int
            Returns zero for successful completion of initialization

        """
        # Enter the cart3d subdirectory
        os.chdir("./cart3d")

        # Touch a file to record the RMS error output for convergence study
        if self.conv_hist:
            with open(self.conv_hist_file, 'w') as f:
                pass

        # Read in the Components.i.tri file
        verts, faces, comps, scalars = ReadTriangulation("Components.i.tri")

        # Store the original tri file location as a class variable
        self.original_tri = os.readlink("Components.i.tri")

        # Store original number of adapt cycles in file
        if self.adapt_growth is not None:
            system_command = r'''awk '{if ("set" == $1 && "n_adapt_cycles" == $2){print $0}}' aero.csh > orig_n_adapt_cycles'''
            os.system(system_command)

        # Extract the relevant components' node locations to body object
        for ibody, body in enumerate(bodies, 1):
            comp_faces = faces[comps == body.id,:]
            comp_verts = np.unique(comp_faces.flatten())

            body.aero_nnodes = len(comp_verts)
            body.aero_X = np.zeros(3*body.aero_nnodes, dtype=TransferScheme.dtype)

            body.aero_X[0::3] = verts[comp_verts,0]
            body.aero_X[1::3] = verts[comp_verts,1]
            body.aero_X[2::3] = verts[comp_verts,2]

            body.rigid_transform = np.identity(4, dtype=TransferScheme.dtype)

            # Initialize the state values used for convergence study as well
            if self.conv_hist:
                self.uprev[body.id] = np.zeros(3*body.aero_nnodes, 
                        dtype=TransferScheme.dtype)
                self.fprev[body.id] = np.zeros(3*body.aero_nnodes, 
                        dtype=TransferScheme.dtype)

        # Head back to run directory
        os.chdir("..")

        return 0

    def get_functions(self,scenario,bodies):
        """
        Populate the scenario with the aerodynamic function values.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies
        """
        pass
        #for function in scenario.functions:
        #    if function.analysis_type=='aerodynamic':
        #        # the [6] index returns the value
        #        if self.comm.Get_rank() == 0:
        #            function.value = interface.design_pull_composite_func(function.id)[6]
        #        function.value = self.comm.bcast(function.value,root=0)

    def iterate(self, scenario, bodies, step):
        """
        Forward iteration of Cart3D

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies.
        step: int
            the time step number

        """
        # Enter the cart3d subdirectory
        os.chdir("./cart3d")

        # Write step number to file output
        if self.conv_hist:
            with open(self.conv_hist_file, 'a') as f:
                f.write("{0:03d} ".format(step))
        
        # Add displacements to node locations and write out to Components.i.tri
        file_in = "Components.i.tri"
        verts, faces, comps, scalars = ReadTriangulation(file_in)

        for ibody, body in enumerate(bodies,1):
            comp_faces = faces[comps == body.id,:]
            comp_verts = np.unique(comp_faces.flatten())

            if 'deform' in body.motion_type:
                verts[comp_verts,:] = body.aero_X.reshape((-1,3)) + \
                                      body.aero_disps.reshape((-1,3))
                
            if 'rigid' in body.motion_type:
                R = body.rigid_transform[:3,:3]
                t = body.rigid_transform[:3,-1]
                verts[comp_verts,:] = vert[comp_verts,:].dot(R.T) + t.T

            # Compute RMS error of displacements and write to file
            if self.conv_hist:
                delta_u_rms = RMS(body.aero_disps, self.uprev[body.id])
                with open(self.conv_hist_file, 'a') as f:
                    f.write("{0:22.15e} ".format(delta_u_rms))

                self.uprev[body.id] = body.aero_disps

        file_out = "Components{0:03d}.i.tri".format(step)
        WriteTri(verts, faces, comps, file_out)

        os.unlink("Components.i.tri")
        os.symlink(file_out, "Components.i.tri")

        # Set the number in adapt cycles according to the growth specified
        if self.adapt_growth is not None:
            # Try to use the number of adapt cycles specified for the current step
            try:
                n_adapt_cycles = self.adapt_growth[step-1]
            # For any steps for which the number isn't specified, repeat last
            except IndexError:
                n_adapt_cycles = self.adapt_growth[-1]

            system_command = r'''awk '{if ("set" == $1 && "n_adapt_cycles" == $2){print $1, $2, $3, ''' + \
                    r'''"{0}"'''.format(n_adapt_cycles) + \
                    r'''}else{print $0}}' aero.csh > aero.csh~'''
            os.system(system_command)
            os.rename("aero.csh~", "aero.csh")
            os.system("chmod +x aero.csh")

        # Run aero.csh to mesh the Components.i.tri and get a flow solution
        os.system("./aero.csh")

        # Read Components.i.triq from BEST directory
        file_in = "BEST/FLOW/Components.i.triq"
        verts, faces, comps, scalars = ReadTriangulation(file_in)

        # Compute the aerodynamic forces
        aero_loads = ComputeAeroLoads(verts, faces, scalars, self.pinf, self.gamma)

        # Pull out the forces from FUN3D
        for ibody, body in enumerate(bodies, 1):
            comp_faces = faces[comps == body.id,:]
            comp_verts = np.unique(comp_faces.flatten())

            body.aero_loads = aero_loads[comp_verts,:].flatten()

            # Compute RMS error of displacements and write to file
            if self.conv_hist:
                delta_f_rms = RMS(body.aero_loads, self.fprev[body.id])
                with open(self.conv_hist_file, 'a') as f:
                    f.write("{0:22.15e} ".format(delta_f_rms))

                self.fprev[body.id] = body.aero_loads

        if self.conv_hist:
            lift = None
            drag = None
            dens_res = None

            # Read loadsCC.dat to get lift and drag
            file_in = "BEST/FLOW/loadsCC.dat"
            try:
                with open(file_in, 'r') as f:
                    data = f.readlines()
                    data = [x.split() for x in data]

                    for line in data:
                        if len(line) > 0 and line[0] == 'entire':
                            if line[1] == 'Lift':
                                lift = float(line[-1])

                            if line[1] == 'Drag':
                                drag = float(line[-1])

            except IOError:
                print("Error: The file " + file_in + " was not found")
                print("Its contents will not appear in convergence history")

            # Read history.dat to get the final global L1 density residual
            file_in = "BEST/FLOW/history.dat"
            try:
                with open(file_in, 'r') as f:
                    data = f.readlines()
                    data = [x.split() for x in data]
                    dens_res = float(data[-1][-1])

            except IOError:
                print("Error: The file " + file_in + " was not found")
                print("Its contents will not appear in convergence history")

            # Write whatever I could get from the files to convergence history
            with open(self.conv_hist_file, 'a') as f:
                if lift is not None:
                    f.write("{0:11.8g} ".format(lift))

                if drag is not None:
                    f.write("{0:11.8g} ".format(drag))

                if dens_res is not None:
                    f.write("{0:11.8e} ".format(dens_res))

                f.write("\n")

        # Make directory to store information from this aeroelastic iteration
        dir_name = "aeroelastic_iteration_{0}".format(step)
        os.system("[ -d {0} ] || mkdir {1}".format(dir_name, dir_name))

        # Move all of the adapt folders into this directory
        os.system("rm -rf {0}/*".format(dir_name))
        os.system("mv -f adapt?? " + dir_name)

        # Copy everything in BEST directory to storage
        #os.system("cp -rf BEST/. {0}".format(dir_name))

        # Use aero_archive.csh to compress other information and store
        #os.system("aero_archive.csh")
        #os.system("mv -f AERO_FILE_ARCHIVE.txt {0}".format(dir_name))
        #os.system("mv -f loadsCC_ARCHIVE.tgz {0}".format(dir_name))

        # Remove adapt folders to prepare for next run
        #os.system("rm -rf adapt??")

        # Head back to run directory
        os.chdir("..")

        return 0

    def post(self, scenario, bodies):
        """
        Clean-up related to running Cart3D
        
        Return the following to their original values:
         - the .tri file that aero.csh uses to generate the mesh
         - the number of adapt cycles specified in aero.csh

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies.

        """
        # Enter the cart3d subdirectory
        os.chdir("./cart3d")

        # Unlink the last tri file and link the original tri file
        os.unlink("Components.i.tri")
        os.symlink(self.original_tri, "Components.i.tri")

        # Read the original number of adapt cycles and replace in aero.csh
        if self.adapt_growth is not None:
            with open("orig_n_adapt_cycles") as f:
                info = f.readline().split()
                n_adapt_cycles = int(info[3])

            system_command = r'''awk '{if ("set" == $1 && "n_adapt_cycles" == $2){print $1, $2, $3, ''' + \
                    r'''"{0}"'''.format(n_adapt_cycles) + \
                    r'''}else{print $0}}' aero.csh > aero.csh~'''
            os.system(system_command)
            os.rename("aero.csh~", "aero.csh")
            os.system("chmod +x aero.csh")

        # Remove any remaining adapt folders to prepare for next run
        os.system("rm -rf adapt??")

        # Head back to run directory
        os.chdir("..")
