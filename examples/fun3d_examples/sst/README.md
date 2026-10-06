# Supersonic Transport Wing (SST)

The Supersonic Transport Wing (SST) is a realistic supersonic aircraft wing geometry (based on the HSCT configuration) used to demonstrate high-fidelity aerothermoelastic optimization with FUN3D, TACS, and CAPS/ESP. The optimization workflow couples aerodynamic heating, structural deformation, and shape design variables to minimize take-off gross weight subject to structural failure constraints at Mach 2.5 cruise conditions.

## Examples

- **[`sst_optimization/`](sst_optimization/)** — Complete multi-step aerothermoelastic optimization workflow. Includes mesh generation, one-way structural sizing, internal structural shape optimization, and fully coupled aeroelastic TOGW minimization with remeshing. See its [README](sst_optimization/README.md) for the full workflow description and run order.

## Reference

Engelstad, S. P., Burke, B. J., Patel, R. N., Sahu, S., and Kennedy, G. J.,
"High-Fidelity Aerothermoelastic Optimization with Differentiable CAD Geometry,"
*AIAA Scitech 2023 Forum*, National Harbor, MD, 2023.
doi:10.2514/6.2023-0329 (<https://doi.org/10.2514/6.2023-0329>)

## Results

Optimized SST wing structural design:

![SST optimized design](results/sst_opt_design.png)

FUN3D flow solution over the SST wing:

![SST FUN3D flow](results/sst_fun3d_flow.png)
