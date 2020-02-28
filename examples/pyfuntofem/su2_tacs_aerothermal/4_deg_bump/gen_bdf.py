import numpy as np

fp = open('plate_bump.bdf', 'w')
fp.write('$ Input file for a curved plate surface\n')
fp.write('SOL 103\nCEND\nBEGIN BULK\n')

# this is changed to create the correct bdf for the 4 deg curve case

yb = []
x = []
b = 0.3048/2
theta = 4.0*np.pi/180.0
r = b/np.sin(theta/2)
for dt in np.linspace(4.0, 0.0, 200)*np.pi/180.0:
    x.append(r*np.cos(np.pi/2 - theta/2 + dt) + r*np.sin(theta/2))
    yb.append(r*np.sin(np.pi/2 - theta/2 + dt) - r*np.cos(theta/2))

ny = 10
y_r = 0.1 #m
y = np.linspace(0.0, y_r, num = ny)

nodes = np.arange(1, len(x)*ny+1, dtype=np.int).reshape(len(x), ny)

# these nodes correspond to a rectangle
# now when writing the grids, check if the node corresponds to the top edge of the rectangle
# if it does, change it to the value of the bump

spclist = []
ct = 0
# Write the grid points to a file
for j in range(ny):
    for i in range(len(x)):
        # Write the nodal data
        spc = ' '
        coord_disp = 0
        coord_id = 0
        seid = 0

        if y[j] == 0.0:
            fp.write('%-8s%16d%16d%16.9e%16.9e*       \n'%
                     ('GRID*', nodes[i, j], coord_id, 
                      x[i], yb[ct]))
            ct += 1
        else:
            fp.write('%-8s%16d%16d%16.9e%16.9e*       \n'%
                     ('GRID*', nodes[i, j], coord_id, 
                      x[i], -1*y[j]))
        fp.write('*       %16.9e%16d%16s%16d        \n'%
                 (0.0, coord_disp, spc, seid))

        if y[j] == 0.1:
            spclist.append(nodes[i,j])

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
    spc = '123456'
    fp.write('%-8s%8d%8d%8s%8.6f\n'%
             ('SPC', 1, node, spc, 350.0))

fp.write('END BULK')
fp.close()
