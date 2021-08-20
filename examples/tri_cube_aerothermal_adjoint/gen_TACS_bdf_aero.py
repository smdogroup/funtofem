import numpy as np

nx = 11
ny = 11

x_min = 0.0
x_max = 5.0 #m
y_min = -5.0
y_max = 0.0 #m

x = np.linspace(x_min, x_max, num = nx)
y = np.linspace(y_min, y_max, num = ny)
theta = np.radians(0.0)
nodes = np.arange(1, nx*ny+1, dtype=np.int).reshape(nx, ny)

fp = open('tacs_aero.bdf', 'w')
fp.write('$ Input file for a rectangular plate\n')
fp.write('SOL 103\nCEND\nBEGIN BULK\n')

spclist = []
spclistT = []
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
                  np.cos(theta)*x[i] - np.sin(theta)*y[j], np.sin(theta)*x[i] + np.cos(theta)*y[j]))
        fp.write('*       %16.9e%16d%16s%16d        \n'%
                 (0.0, coord_disp, spc, seid))

        if j == 0 and (x[i] == x_min or x[i] == x_max) : #(ny-1)/2
            spclist.append(nodes[i,j])

        if y[j] == y_min: #x[i] == x_min or x[i] == x_max:#y[j] == y_min:
            spclistT.append(nodes[i,j])

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

for node in spclist:
    spc = '123'
    fp.write('%-8s%8d%8d%8s%8.6f\n'%
             ('SPC', 1, node, spc, 0.0))

for node in spclistT:
    spc = '123456'#6
    fp.write('%-8s%8d%8d%8s%8.4f\n'%
             ('SPC', 1, node, spc, 300))

fp.write('END BULK')
fp.close()
