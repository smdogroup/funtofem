# BUCKLING CONSTRAINTS
# -----------------------------------------------------
thick = (
    Variable.structural("thick", value=1.0)
    .set_bounds(lower=0.1, upper=10.0, scale=10.0)
    .register_to(wing)
)
T = Function.temperature().register_to(cruise)

# define material properties here (assuming metal / isotropic)
E = aluminum._E1
Tref = T_ref  # K maybe?
nu = aluminum._nu12
alpha = aluminum._alpha1  # CTE (coeff thermal expansion)

# plate dimensions
a = 5.0 / (nribs + 1)  # 1-direction length
b = 1.0 / (nspars + 1)  # 2-direction width

# stiffener properties
num_stiff = 0
stiff_height = 0
stiff_thick = 0

N = num_stiff + 1
s_p = b / N  # spar pitch

# compute thermal stress and in-plane loads assuming plate is pinned
# on all sides so no axial contraction (constrained)
dT = T - Tref
sigma_11 = (
    alpha * dT * E / (1 - nu)
)  # compressive thermal stress here (+ is compressive)
N11 = sigma_11 * thick

# compute some laminate plate properties (despite metal and non-laminate)
Q11 = E / (1 - nu**2)
Q22 = Q11
Q12 = nu * Q11
G12 = E / 2.0 / (1 + nu)
Q66 = G12
I = thick**3 / 12.0
D11 = Q11 * I
D22 = Q22 * I
D12 = Q12 * I
D66 = G12 * I

# compute important non-dimensional parameters
rho_0 = a / b
# xi normally defined with D but only one-ply metal so simplify with floats only (not CompositeFunctions)
xi = (Q12 + 2 * Q66) / (Q11 * Q22) ** 0.5

# compute stiffness ratio
I_stiff = stiff_thick * stiff_height**3 / 12.0
A_stiff = stiff_thick * stiff_height
A_plate = b * thick
z_wall = (stiff_height + thick) / 2
z_cen = A_stiff * z_wall * num_stiff / (A_stiff * num_stiff + A_plate)
EI_s = E * (I_stiff + A_stiff * (z_cen - z_wall) ** 2)
gamma = EI_s / s_p / D11

# if stiffener ratio gamma isn't too large (gamma < 1), TODO : please check this
print(f"gamma = {gamma}")
# then you can assume m1 close to rho_0 the plate aspect ratio
m1 = int(rho_0)

# compute the critical in-plane load
Dgeom_avg = D11  # would be sqrt(D11 * D22) but isotropic these are equal
N11_cr = (
    np.pi**2
    * Dgeom_avg
    / b**2
    * ((1 + gamma) * m1**2 / rho_0**2 + rho_0**2 / m1**2 + 2 * xi)
)

# compute the buckling failure criterion
safety_factor = 1.5
mu_thermal_buckle = N11 / N11_cr * safety_factor
mu_thermal_buckle.set_name("mu_thermal_buckle").optimize(
    upper=1.0, scale=1e0, objective=False, plot=True
).register_to(f2f_model)
