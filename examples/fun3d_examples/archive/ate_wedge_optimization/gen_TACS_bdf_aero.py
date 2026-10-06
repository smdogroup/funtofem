import numpy as np

nx = 11
ny = 5
nz = 5

x_min = 1.0
x_max = 2.0  # m
y_min = -0.5
y_max = 0.5  # m

z_min = -0.015
z_max = -0.005

x = np.linspace(x_min, x_max, num=nx)
y = np.linspace(y_min, y_max, num=ny)
z = np.linspace(z_min, z_max, num=nz)
theta = np.radians(5.0)
nodes = np.arange(1, nx * ny * nz + 1, dtype=np.int).reshape(nx, ny, nz)

fp = open("tacs_aero.bdf", "w")
fp.write("$ Input file for a rectangular plate\n")
fp.write("SOL 103\nCEND\nBEGIN BULK\n")

spclist = []
spclistY = []
spclistT = []
# Write the grid points to a file
for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            # Write the nodal data
            spc = " "
            coord_disp = 0
            coord_id = 0
            seid = 0

            xpt = np.cos(theta) * x[i] - np.sin(theta) * z[k] + 1.0
            ypt = y[j]
            zpt = np.sin(theta) * x[i] + np.cos(theta) * z[k]

            fp.write(
                "%-8s%16d%16d%16.9e%16.9e*       \n"
                % ("GRID*", nodes[i, j, k], coord_id, xpt, ypt)
            )
            fp.write(
                "*       %16.9e%16d%16s%16d        \n" % (zpt, coord_disp, spc, seid)
            )

            # If the node is on one of the bottom outer edges of the plate,
            # restrain it against displacement in any direction
            if k == 0 and (i == 0 or i == nx - 1):
                spclist.append(nodes[i, j, k])

            # If the node is on one of the outer edges of the plate,
            # restrain it against displacement in y direction
            if k != 0 and (j == 0 or j == ny - 1):
                spclistY.append(nodes[i, j, k])

            # Set the temperature along the bottom edge of the plate
            if k == 0:
                spclistT.append(nodes[i, j, k])

# Write out the linear hexahedral elements
elem = 1
for k in range(0, nodes.shape[2] - 1, 1):
    for j in range(0, nodes.shape[1] - 1, 1):
        for i in range(0, nodes.shape[0] - 1, 1):
            # Set different part numbers for the elements on the
            # lower and volume mesh
            part_id = i + 1
            if k == 0:
                part_id = i + nodes.shape[0]
            # Write the connectivity data
            fp.write(
                "%-8s%8d%8d%8d%8d%8d%8d%8d%8d\n%-8s%8d%8d\n"
                % (
                    "CHEXA",
                    elem,
                    part_id,
                    nodes[i, j, k],
                    nodes[i + 1, j, k],
                    nodes[i + 1, j + 1, k],
                    nodes[i, j + 1, k],
                    nodes[i, j, k + 1],
                    nodes[i + 1, j, k + 1],
                    "*",
                    nodes[i + 1, j + 1, k + 1],
                    nodes[i, j + 1, k + 1],
                )
            )
            elem += 1

# # Add an extra layer of hexa elements and add the tractions through the
# # TACSTraction3D class.
# part_id = 2
# k0 = 0
# for j in range(0, nodes.shape[1]-1, 1):
#     for i in range(0, nodes.shape[0]-1, 1):
#         # Write the connectivity data
#         fp.write('%-8s%8d%8d%8d%8d%8d%8d%8d%8d\n%-8s%8d%8d\n'%
#                     ('CHEXA', elem, part_id,
#                     nodes[i, j, k0], nodes[i+1, j ,k0],
#                     nodes[i+1, j+1, k0], nodes[i, j+1, k0],
#                     nodes[i, j, k0+1], nodes[i+1, j, k0+1], '*',
#                     nodes[i+1, j+1, k0+1], nodes[i, j+1, k0+1]))
#         elem += 1

# Note: This approach will not work because the CQUAD4 pressure elements are not
# yet implemented in TACS.
#
# # Write out the elements associated with the back-pressure. These are quads and
# # are only associated with the back side of the structure
# k0 = 0 # Set the plane of nodes where we will apply the surface traction
# part_id = 2 # Set a different part id
# for i in range(0, nodes.shape[0]-1, 1):
#     for j in range(0, nodes.shape[1]-1, 1):
#         n = [nodes[i, j, k0], nodes[i+1, j, k0],
#              nodes[i+1, j+1, k0], nodes[i, j+1, k0]]
#         # Note that the orientation of the element is reversed so that the normal
#         # points in the -z direction
#         fp.write('%-8s%8d%8d%8d%8d%8d%8d\n'%
#                  ('CQUAD4', elem, part_id, n[0], n[3], n[2], n[1]))
#         elem += 1

for node in spclist:
    spc = "123"
    fp.write("%-8s%8d%8d%8s%8.6f\n" % ("SPC", 1, node, spc, 0.0))

for node in spclistY:
    spc = "2"
    fp.write("%-8s%8d%8d%8s%8.6f\n" % ("SPC", 1, node, spc, 0.0))

for node in spclistT:
    spc = "4"
    fp.write("%-8s%8d%8d%8s%8.4f\n" % ("SPC", 1, node, spc, 300.0))

fp.write("END BULK")
fp.close()
