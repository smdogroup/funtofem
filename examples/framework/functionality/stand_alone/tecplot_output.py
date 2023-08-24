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

"""
--------------------------------------------------------------------------------
Tecplot output
--------------------------------------------------------------------------------
The functionality provided in this script duplicates the output file-writing
routine in FUNtoFEM's source code. It was duplicated so that this version could
be modified at the convenience of the developer or user instead of the version
in the C++ source. The one in source should only be used for debugging purposes.
"""
import numpy as np


def writeOutputForTecplot(
    struct_X,
    aero_X,
    struct_disps,
    aero_disps,
    struct_conn=np.array([]),
    aero_conn=np.array([]),
    struct_ptr=np.array([]),
    aero_ptr=np.array([]),
    struct_elem_type=0,
    aero_elem_type=0,
    struct_loads=np.array([]),
    aero_loads=np.array([]),
    filename="funtofem_output.dat",
):
    """
    Writes node locations and connectivity of structural and aerodynamic surface
    meshes to file that can be read by Tecplot

    Arguments:
    ----------
    struct_X -- 1D array of structural node locations
    aero_X -- 1D array of aerodynamic surface node locations
    struct_conn -- connectivity of structural mesh
    aero_conn -- connectivity of aerodynamic surface mesh
    struct_ptr -- array of indices in struct_conn corresponding to each elem
    aero_ptr -- array containing indices of conn corresponding to each elem
    struct_elem_type -- element type (1 for tris, 2 for quads)
    aero_elem_type -- element type (1 for tris, 2 for quads)
    struct_loads -- 1D array of loads on structural nodes
    aero_loads -- 1D array of loads on aerodynamic surface nodes
    filename -- name of output file
    """
    struct_nnodes = len(struct_X) // 3
    aero_nnodes = len(aero_X) // 3

    struct_conn_provided = struct_conn.size and struct_ptr.size
    aero_conn_provided = aero_conn.size and aero_ptr.size

    struct_zone_str = "\n"
    if struct_elem_type == 1:
        struct_zone_str = "zonetype=fetriangle datapacking=point\n"
    elif struct_elem_type == 2:
        struct_zone_str = "zonetype=fequadrilateral datapacking=point\n"

    aero_zone_str = "\n"
    if aero_elem_type == 1:
        aero_zone_str = "zonetype=fetriangle datapacking=point\n"
    elif aero_elem_type == 2:
        aero_zone_str = "zonetype=fequadrilateral datapacking=point\n"

    loads_provided = struct_loads.size and aero_loads.size
    variables_str = 'VARIABLES="x", "y", "z"\n'
    if loads_provided:
        variables_str = 'VARIABLES="x", "y", "z", "fx", "fy", "fz"\n'

    f = open(filename, "w")

    # Write Tecplot compatible header
    f.write('TITLE="FUNtoFEM output"\n')
    f.write(variables_str)

    # Write Tecplot zone information for the undeformed structural mesh
    f.write('ZONE T="Structural mesh"\n')
    f.write("I=%i " % struct_nnodes)
    f.write(struct_zone_str)

    # Write structural node locations, connectivity, and loads
    for j in range(struct_nnodes):
        f.write(
            "%22.15e %22.15e %22.15e "
            % (
                struct_X[3 * j + 0].real,
                struct_X[3 * j + 1].real,
                struct_X[3 * j + 2].real,
            )
        )
        if loads_provided:
            f.write(
                "%22.15e %22.15e %22.15e"
                % (
                    struct_loads[3 * j + 0].real,
                    struct_loads[3 * j + 1].real,
                    struct_loads[3 * j + 2].real,
                )
            )
        f.write("\n")
    if struct_conn_provided:
        for m in range(len(struct_ptr) - 1):
            for n in range(struct_ptr[m], struct_ptr[m + 1]):
                f.write("%i " % struct_conn[n])
            f.write("\n")

    # Write Tecplot zone information for the undeformed aerodynamic surface mesh
    f.write('ZONE T="Aerodynamic surface mesh"\n')
    f.write("I=%i " % aero_nnodes)
    f.write(aero_zone_str)

    # Write aerodynamic surface node locations, connectivity, and loads
    for i in range(aero_nnodes):
        f.write(
            "%22.15e %22.15e %22.15e "
            % (aero_X[3 * i + 0].real, aero_X[3 * i + 1].real, aero_X[3 * i + 2].real)
        )
        if loads_provided:
            f.write(
                "%22.15e %22.15e %22.15e"
                % (
                    aero_loads[3 * i + 0].real,
                    aero_loads[3 * i + 1].real,
                    aero_loads[3 * i + 2].real,
                )
            )
        f.write("\n")
    if aero_conn_provided:
        for m in range(len(aero_ptr) - 1):
            for n in range(aero_ptr[m], aero_ptr[m + 1]):
                f.write("%i " % aero_conn[n])
            f.write("\n")

    # Write Tecplot zone information for the deformed structural mesh
    f.write('ZONE T="Deformed structural mesh"\n')
    f.write("I=%i " % struct_nnodes)
    f.write(struct_zone_str)

    # Write deformed structural node locations and connectivity
    for j in range(struct_nnodes):
        f.write(
            "%22.15e %22.15e %22.15e "
            % (
                struct_X[3 * j + 0].real + struct_disps[3 * j + 0].real,
                struct_X[3 * j + 1].real + struct_disps[3 * j + 1].real,
                struct_X[3 * j + 2].real + struct_disps[3 * j + 2].real,
            )
        )
        if loads_provided:
            f.write("%22.15e %22.15e %22.15e" % (0.0, 0.0, 0.0))
        f.write("\n")
    if struct_conn_provided:
        for m in range(len(struct_ptr) - 1):
            for n in range(struct_ptr[m], struct_ptr[m + 1]):
                f.write("%i " % struct_conn[n])
            f.write("\n")

    # Write Tecplot zone information for the deformed aerodynamic surface mesh
    f.write('ZONE T="Deformed aerodynamic surface mesh"\n')
    f.write("I=%i " % aero_nnodes)
    f.write(aero_zone_str)

    # Write deformed aerodynamic surface node locations and connectivity
    for i in range(aero_nnodes):
        f.write(
            "%22.15e %22.15e %22.15e "
            % (
                aero_X[3 * i + 0].real + aero_disps[3 * i + 0].real,
                aero_X[3 * i + 1].real + aero_disps[3 * i + 1].real,
                aero_X[3 * i + 2].real + aero_disps[3 * i + 2].real,
            )
        )
        if loads_provided:
            f.write("%22.15e %22.15e %22.15e" % (0.0, 0.0, 0.0))
        f.write("\n")
    if aero_conn_provided:
        for m in range(len(aero_ptr) - 1):
            for n in range(aero_ptr[m], aero_ptr[m + 1]):
                f.write("%i " % aero_conn[n])
            f.write("\n")

    f.close()
