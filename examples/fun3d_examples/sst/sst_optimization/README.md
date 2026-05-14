# Supersonic Transport Wing (SST) — Full Optimization Example

This directory contains a multi-step aerothermoelastic optimization of the SST (HSCT) aircraft wing, adapted from the research case in Engelstad et al. (2023). The workflow progresses from one-way structural sizing through fully coupled aeroelastic and aerothermoelastic optimization, using FUN3D for CFD, TACS for structural analysis, and CAPS/ESP for geometry parameterization and mesh generation.

## Flight Conditions (Cruise)

| Parameter | Value |
|-----------|-------|
| Altitude | 60,000 ft |
| Mach number | Mach 2.5 |
| T∞ | 216 K |
| ρ∞ | 0.1165 kg/m³ |
| Speed of sound | 295 m/s |
| V∞ | 737.5 m/s |
| q∞ | 3.1682×10⁴ Pa |
| Re | 7.3817×10⁸ |
| y⁺ | 15 (turbulent wall spacing = 1×10⁻⁴ m) |

## Dependencies

- **FUN3D** — CFD solver
- **TACS** — structural finite element solver
- **CAPS/ESP (pyCAPS)** — geometry parameterization and mesh generation
- **mpi4py** — MPI parallelism
- **pyoptsparse (SNOPT)** — gradient-based optimization

## Directory Layout

```
sst_optimization_full/
├── cfd/        # FUN3D mesh files and CAPS fluid analysis outputs
├── geometry/   # CSM geometry source file (sst_v2.csm)
├── struct/     # TACS structural mesh files and CAPS structural analysis outputs
├── design/     # Optimization history files (.hst) and design variable files (.txt)
├── _mesh_fun3d.py
├── _mesh_tacs.py
├── _run_flow.py
├── _run_inviscid_ae.py
├── 1_sizing_optimization.py
├── 2_sizing_shape.py
├── 3_eval_inviscid.py
├── 4_eval_turb.py
├── 5_fc_inviscid_ae_remesh.py
└── 6_fc_inviscid_der_test.py
```

## Helper Scripts

These scripts handle mesh generation and standalone forward analyses that are prerequisites for the numbered optimization scripts.

### `_mesh_fun3d.py`

Generates the FUN3D volume mesh via CAPS/ESP AFLR. Supports both inviscid and turbulent boundary conditions (controlled by a flag at the top of the script). Outputs the mesh files to `cfd/`.

### `_mesh_tacs.py`

Generates the TACS structural mesh via CAPS/ESP EGADS. Outputs the structural mesh files to `struct/`.

### `_run_flow.py`

Runs a one-way aero-only forward analysis (no structural coupling) to produce an aero loads file. The loads file is written to `cfd/` and is used as input to the one-way structural sizing scripts (1 and 2).

### `_run_inviscid_ae.py`

Runs a fully coupled inviscid aeroelastic forward analysis. This script is called as a subprocess by scripts 5 and 6 via the FUNtoFEM `Remote` driver interface.

## Numbered Scripts

The numbered scripts form a progressive optimization workflow, each building on the results of the previous step.

### `1_sizing_optimization.py`

**Purpose**: One-way structural sizing optimization. Minimizes wing structural mass subject to a KS failure constraint (ksfailure ≤ 1).

**Inputs**: Aero loads file produced by `_run_flow.py` (in `cfd/`).

**Outputs**: Optimal panel thicknesses written to `design/sizing.txt`.

---

### `2_sizing_shape.py`

**Purpose**: One-way sizing plus internal structural shape optimization. Extends script 1 by adding rib and spar orientation angles as shape design variables.

**Inputs**: Aero loads file from `cfd/`; initial design from `design/sizing.txt` (optional).

**Outputs**: Optimal design (thicknesses + shape variables) written to `design/internal-struct.txt`.

---

### `3_eval_inviscid.py`

**Purpose**: Fully coupled inviscid aeroelastic forward analysis (evaluation only, no optimization). Useful for verifying the design from script 2 under coupled aero-structural loading.

**Inputs**: Design variables from `design/internal-struct.txt`.

**Outputs**: Prints all functional values (mass, KS failure, lift, drag) to stdout.

---

### `4_eval_turb.py`

**Purpose**: Fully coupled turbulent aerothermoelastic forward analysis (evaluation only). Evaluates the final design under turbulent flow with aerodynamic heating.

**Inputs**: Design variables from `design/inviscid-ae.txt` (produced by script 5).

**Outputs**: Prints all functional values to stdout.

---

### `5_fc_inviscid_ae_remesh.py`

**Purpose**: Fully coupled inviscid aeroelastic TOGW (take-off gross weight) minimization with geometry remeshing at each major iteration. This is the primary coupled optimization script.

**Inputs**: Initial design from `design/internal-struct.txt`. Calls `_run_inviscid_ae.py` as a subprocess via the `Remote` driver.

**Outputs**: Optimized design written to `design/inviscid-ae.txt`.

---

### `6_fc_inviscid_der_test.py`

**Purpose**: Finite-difference derivative test for the fully coupled inviscid driver. Validates that adjoint-computed gradients match finite-difference approximations to the specified tolerance.

**Inputs**: Design variables from `design/internal-struct.txt`. Calls `_run_inviscid_ae.py` as a subprocess.

**Outputs**: Prints gradient comparison results to stdout.

## Recommended Run Order

```
_mesh_fun3d.py → _mesh_tacs.py → _run_flow.py → 1_sizing_optimization.py
    → 2_sizing_shape.py → 3_eval_inviscid.py → 5_fc_inviscid_ae_remesh.py → 4_eval_turb.py
```

Script 6 (`6_fc_inviscid_der_test.py`) can be run after script 2 to validate gradients before launching the full coupled optimization in script 5.

## Citation

```
Engelstad, S. P., Burke, B. J., Patel, R. N., Sahu, S., and Kennedy, G. J.,
"High-Fidelity Aerothermoelastic Optimization with Differentiable CAD Geometry,"
AIAA Scitech 2023 Forum, National Harbor, MD, 2023. doi:10.2514/6.2023-0329.
```
