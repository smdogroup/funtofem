# FUN3D Examples #

* `ate_wedge_optimization` - Optimization of a supersonic panel under supersonic flow using with aerothermoelastic analysis. Uses <font color="blue">FUN3D</font>, <font color="orange">TACS</font>, <font color="green">MELD</font>, `FuntofemNlbgs` driver.
* `diamond_unsteady` - Unsteady forward and adjoint analysis of a diamond wedge wing structure under aeroelastic analysis. Uses <font color="blue">FUN3D</font>, <font color="orange">TACS</font>, <font color="green">MELD</font>.
* `pyopt_togw_optimization` - Optimization of the CRM  aircraft structure under aeroelastic analysis. Uses <font color="blue">FUN3D</font>, <font color="orange">TACS</font>, <font color="green">MELD</font>, `FuntofemNlbgs` driver.
* `sst_optimization` - Optimization of a supersonic transport wing under aerothermoelastic analysis using ksfailure and mass. Uses <font color="blue">FUN3D</font>, <font color="orange">TACS</font>, <font color="green">MELD</font>, `FuntofemNlbgs` driver.
* `sst_unsteady` - Unsteady forward and adjoint analysis of a simplified supersonic transport wing geometry. Uses <font color="blue">FUN3D</font>, <font color="orange">TACS</font>, <font color="green">MELD</font>.

### Supersonic Transport Wing ###
<i>Directory</i> - `sst_optimization`
The supersonic transport wing was the first demonstration of aerothermoelastic analysis with FUN3D and TACS on a realistic aircraft structure, included in the following paper. 
```r
Engelstad, S. P., Burke, B. J., Patel, R. N., Sahu, S., and Kennedy, G. J., “High-Fidelity Aerothermoelastic Optimization with
Differentiable CAD Geometry,” AIAA Scitech 2023 Forum, National Harbor, MD, 2023. doi:10.2514/6.2023-0329.
```
<figure class="image">
  <img src="sst_optimization/results/sst_opt_design.png" width=600 />
  <figcaption><em>Optimal thicknesses for the supersonic transport wing.</em></figcaption>
</figure>
<figure class="image">
  <img src="sst_optimization/results/sst_fun3d_flow.png" width=500 />
  <figcaption><em>Pressure contours in the Mach 2.0 flow solved in FUN3D.</em></figcaption>
</figure>

### Computational Research Model ###
<i>Directory</i> - `pyopt_togw_optimization`
```r
Jacobson, K., Kiviaho, J., Smith, M., and Kennedy, G., “An Aeroelastic Coupling Framework for Time-accurate Anal-
ysis and Optimization,” 2018 AIAA/ASCE/AHS/ASC Structures, Structural Dynamics, and Materials Conference, 2018.
doi:10.2514/6.2018-0100.
```
<img src="pyopt_togw_optimization/images/crm_thick_opt.png" width=500 />