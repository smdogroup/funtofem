"""
Sean Engelstad, Nov 2023
GT SMDO Lab, Dr. Graeme Kennedy
OnewayStructDriver example

Requires install of FUNtoFEM, and TACS
"""

from funtofem import *
import os
from mpi4py import MPI
import matplotlib.pyplot as plt

# --------------------------------------------------------------#
# Setup CAPS Problem and FUNtoFEM model
# --------------------------------------------------------------#
comm = MPI.COMM_WORLD
f2f_model = FUNtoFEMmodel("tacs_sphere")
sphere = Body.aeroelastic(
    "sphere"
)  # says aeroelastic but not coupled, may want to make new classmethods later...

# register the funtofem Body to the model
sphere.register_to(f2f_model)

# make the scenario(s)
tacs_scenario = Scenario.steady("tacs", steps=1)
tacs_scenario.register_to(f2f_model)

# generate aerodynamic surface mesh of a sphere
n1 = 100; m1 = 100
rho = 1
theta = np.linspace(0,2*np.pi,n1)
psi = np.linspace(-np.pi/2, np.pi/2, m1)
TH_a, PSI_a = np.meshgrid(theta, psi)
Xa = rho*np.cos(TH_a)*np.cos(PSI_a)
Ya = rho*np.sin(TH_a)*np.cos(PSI_a)
Za = rho*np.sin(PSI_a)

# plot the 3D sphere => aerodynamic surface mesh
fig = plt.figure("aero_sphere")
ax = fig.add_subplot(111, projection='3d')
my_col = plt.cm.jet(Za/np.amax(Za))
ax.plot_surface(Xa, Ya, Za, facecolors=my_col, antialiased=False, linewidth=2)
plt.savefig("aero-surface-mesh.png", dpi=300)
plt.close("aero_sphere")

# generate struct surface mesh of a sphere
n2 = 30; m2 = 30
rho = 1
theta = np.linspace(0,2*np.pi,n2)
# slight angle offset here to make sure no nodes match the aero surf mesh
theta = np.array([(th+0.2312)%(2*np.pi) for th in theta])
psi = np.linspace(-np.pi/2, np.pi/2, m2)
TH_s, PSI_s = np.meshgrid(theta, psi)
Xs = rho*np.cos(TH_s)*np.cos(PSI_s)
Ys = rho*np.sin(TH_s)*np.cos(PSI_s)
Zs = rho*np.sin(PSI_s)

# plot the 3D sphere => aerodynamic surface mesh
fig = plt.figure("struct_sphere")
ax = fig.add_subplot(111, projection='3d')
my_col = plt.cm.jet(Zs/np.amax(Zs))
ax.plot_surface(Xs, Ys, Zs, facecolors=my_col, antialiased=False)
plt.savefig("struct-surface-mesh.png", dpi=300)
plt.close("struct_sphere")

# collect into the aero_X list, [x1,y1,z1,x2,y2,z2,...]
aero_X = np.zeros((3*n1*m1))
for i in range(n1*m1):
    row = i % n1; col = int(np.floor(i/n1))
    aero_X[3*i] = Xa[row,col]
    aero_X[3*i+1] = Ya[row,col]
    aero_X[3*i+2] = Za[row,col]
aero_id = np.arange(1, n1*m1+1)
sphere.initialize_aero_nodes(aero_X, aero_id)

# collect into the struct_X list, [x1,y1,z1,x2,y2,z2,...]
struct_X = np.zeros((3*n2*m2))
for i in range(n2*m2):
    row = i % n2; col = int(np.floor(i/n2))
    struct_X[3*i] = Xs[row,col]
    struct_X[3*i+1] = Ys[row,col]
    struct_X[3*i+2] = Zs[row,col]
struct_id = np.arange(1, n2*m2+1)
sphere.initialize_struct_nodes(struct_X, struct_id)

# verify aero mesh
print(f"aero X, shape {aero_X.shape} = {sphere.aero_X}")
# verify struct mesh
print(f"struct X, shape {struct_X.shape} = {sphere.struct_X}")

# transfer scheme settings, sym about Y axis
char_dist = 1
beta = 1.0/char_dist**2
m_transfer_settings = TransferSettings(elastic_scheme="meld", npts=50, beta=beta, isym=-1)

# initialize transfer scheme
sphere.initialize_transfer(comm, struct_comm=comm, struct_root=0, aero_comm=comm, aero_root=0, transfer_settings=m_transfer_settings)
sphere.initialize_variables(tacs_scenario)

# generate random struct disps to transfer, and set into body
npts_struct = sphere.get_num_struct_nodes()
# choose meaningful struct disps in this case u = 0.5 * x^3
#[u1, v1, w1, u2, v2, w2, ...]
struct_disps = np.zeros((3*n2*m2,))
# NOTE : you can change the displacement field here
for i in range(n2*m2):
    row = i % n2; col = int(np.floor(i/n2))
    struct_disps[3*i] = -0.5 * Xs[row,col]**3
    struct_disps[3*i+1] = -0.5 * Ys[row,col]**3
    struct_disps[3*i+2] = 0.0
sphere.struct_disps[tacs_scenario.id] = struct_disps

# generate random aero loads to transfer, and set into body
aero_loads = np.random.rand(3*n1*m1) #[fx1, fy1, fz1, fx2, fy2, fz2,...]
sphere.aero_loads[tacs_scenario.id] = aero_loads

# get the MELD transfer object if you want (directly)
# meld_transfer = sphere.transfer

# transfer struct to aero disps uS => uA
sphere.transfer_disps(tacs_scenario, time_index=0)

# transfer aero loads to struct loads fA => fS
sphere.transfer_loads(tacs_scenario, time_index=0)

# get the aero disps
aero_disps = sphere.get_aero_disps(tacs_scenario, time_index=0)

# get the struct loads
# copy this before doing structural analysis
struct_loads = sphere.get_struct_loads(tacs_scenario, time_index=0) * 1.0
print(f"struct loads = {struct_loads}")

# then from here you can use the uS, uA, fS, fA to plot it on the plate and/or sphere
# and check consistency of the transfer scheme.

# NOTE : you probably want to compute work or something here, but this is a start.
print(f"struct disp norm = {np.linalg.norm(struct_disps)}")
print(f"aero disp norm = {np.linalg.norm(aero_disps)}")
print(f"aero loads norm = {np.linalg.norm(aero_loads)}")
print(f"struct loads norm = {np.linalg.norm(struct_loads)}")

# visualize the aero disps on the sphere, with scale factor
Xad = Xa * 1.0
Yad = Ya * 1.0
Zad = Za * 1.0 # copy X,Y,Z aero mesh for deformed version
scale = 1.0
for i in range(n1*m1):
    row = i % n1; col = int(np.floor(i/n1))
    Xad[row,col] = Xa[row,col] + scale * aero_disps[3*i]
    Yad[row,col] = Ya[row,col] + scale * aero_disps[3*i+1]
    Zad[row,col] = Za[row,col] + scale * aero_disps[3*i+2]
fig = plt.figure("aero_sphere_def")
ax = fig.add_subplot(111, projection='3d')
my_col = plt.cm.jet(Zad/np.amax(Zad))
ax.plot_surface(Xad, Yad, Zad, facecolors=my_col, antialiased=False, linewidth=2)
ax.set_aspect('equal', adjustable='box')
plt.xlim(-1.0, 1.0)
plt.ylim(-1.0, 1.0)
#plt.zlim(-1.0, 1.0)
plt.savefig("aero-surface-def.png", dpi=300)
plt.close("aero_sphere_def")

# visualize the struct disps on the sphere, with scale factor
Xsd = Xs * 1.0
Ysd = Ys * 1.0
Zsd = Zs * 1.0 # copy X,Y,Z aero mesh for deformed version
scale = 1.0
for i in range(n2*m2):
    row = i % n2; col = int(np.floor(i/n2))
    Xsd[row,col] = Xs[row,col] + scale * struct_disps[3*i]
    Ysd[row,col] = Ys[row,col] + scale * struct_disps[3*i+1]
    Zsd[row,col] = Zs[row,col] + scale * struct_disps[3*i+2]
fig = plt.figure("struct_sphere_def")
ax = fig.add_subplot(111, projection='3d')
my_col = plt.cm.jet(Zsd/np.amax(Zsd))
ax.plot_surface(Xsd, Ysd, Zsd, facecolors=my_col, antialiased=False, linewidth=2)
ax.set_aspect('equal', adjustable='box')
plt.xlim(-1.0, 1.0)
plt.ylim(-1.0, 1.0)
#plt.zlim(-1.0, 1.0)
plt.savefig("struct-surface-def.png", dpi=300)
plt.close("struct_sphere_def")

# compare the work from loads on the aero and struct mesh (conservation)
aero_work = np.dot(aero_disps, aero_loads)
struct_work = np.dot(struct_disps, struct_loads)
print(f"aero work = {aero_work}")
print(f"struct work = {struct_work}")

# compare total loads in each direction
for d in range(3):
    fA = np.sum(aero_loads[d::3])
    fS = np.sum(struct_loads[d::3])
    print(f"fA,{d} = {fA}")
    print(f"fS,{d} = {fS}")