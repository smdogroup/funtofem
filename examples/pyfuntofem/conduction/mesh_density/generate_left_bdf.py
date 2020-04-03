import numpy as np

nx = 101
ny = 101

x = np.linspace(0, 1.0, num = nx)
y = np.linspace(0, 1.0, num = ny)

nodes = np.arange(1, nx*ny+1, dtype=np.int).reshape(nx, ny)

fp = open('left_plate.bdf', 'w')
fp.write('$ Input file for the left plate\n')
fp.write('SOL 103\nCEND\nBEGIN BULK\n')

spclist1 = []
spclist2 = []

# Write the grid points to a file
for j in range(ny):
    for i in range(nx):
        # Write the nodal data
        spc = ' '
        coord_disp = 0
        coord_id = 0
        seid = 0
        
        fp.write('%-8s%16d%16d%16.9e%16.9e*       \n'%
                 ('GRID*', nodes[i, j], coord_id, 
                  x[i], y[j]))
        fp.write('*       %16.9e%16d%16s%16d        \n'%
                 (0.0, coord_disp, spc, seid))

        if x[i] == 0.0:
            spclist1.append(nodes[i,j])

        if x[i] == 1.0:
            spclist2.append(nodes[i,j])

# Output first order quad elements

elem = 1
part_id = 1
for j in range(0, nodes.shape[1]-1, 1):
    for i in range(0, nodes.shape[0]-1, 1):
        # Write the connectivity data
        # CQUAD9 elem id n1 n2 n3 n4 n5 n6
        #        n7   n8 n9
        fp.write('%-8s%8d%8d%8d%8d%8d%8d\n'%
                 ('CQUAD4', elem, part_id, 
                  nodes[i,   j],   nodes[i+1, j], 
                  nodes[i+1, j+1], nodes[i,   j+1]))

        elem += 1

for node in spclist1:
    spc = '123456'
    fp.write('%-8s%8d%8d%8s%8.6f\n'%
             ('SPC', 1, node, spc, 300.0))

for node in spclist2:
    spc = '123456'
    fp.write('%-8s%8d%8d%8s%8.6f\n'%
             ('SPC', 1, node, spc, 400.0))

fp.write('END BULK')
fp.close()
