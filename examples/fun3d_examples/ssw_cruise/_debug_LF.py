clift = 0.95e-1  # at AOA 2.0 deg, sym airfoil
# improved lift coeff with a better airfoil
# adjusted with a multiplier (will shape optimize for this later)
clift *= 14.0  # 0.095 => 1.33 approx
mass_wingbox = 308  # kg
q_inf = 1.21945e4
# flying wing, glider structure
mass_payload = 100  # kg
mass_frame = 0  # kg
mass_fuel_res = 2e3  # kg
LGM = mass_payload + mass_frame + mass_fuel_res + 2 * mass_wingbox
print(f"LGM = {LGM}, wing frac {2*mass_wingbox / LGM}")
LGW = 9.81 * LGM  # kg => N
dim_lift = clift * 2 * q_inf
print(f"dim lift = {dim_lift}")
print(f"LGW = {LGW}")
load_factor = dim_lift - LGW
rel_err = load_factor / dim_lift
print(f"load factor = {load_factor}, rel err = {rel_err}")
