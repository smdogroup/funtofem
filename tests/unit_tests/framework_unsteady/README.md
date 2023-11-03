# Unsteady Framework Tests #
* `test_framework_adjoint_eqns.py` - hardcode the unsteady adjoint equation matrix and total derivatives for two unsteady steps, to verify the unsteady framework.
* `test_framework_unsteady.py` - Test the fully-coupled unsteady analysis with the TestAero+TestStruct solvers.
* `test_tacs_interface_unsteady.py` - Test the structural and aerodynamic derivatives of a fully-coupled unsteady analysis with TestAero + TACS solvers using default f2f callback and _mesh_plate.py mesh.
* `test_tacs_interface_unsteady.py` - Test the structural and aerodynamic derivatives of a fully-coupled unsteady analysis with TestAero + TACS solvers.
* `test_unsteady_solvers.py` - directional derivative test for the two test aero + test struct solvers for unsteady.

## Unsteady Aeroelastic Analysis
The unsteady aeroelastic forward analysis involves displacements $u_{\color{blue}{A}}^i, u_{\color{orange}{S}}^i$ and forces $f_{\color{blue}{A}}^i, f_{\color{orange}{S}}^i$ for each time step $i$.
```math
\begin{aligned}
u_{\color{orange}{S}}^0 \\
u_{\color{blue}{A}}^1 \rightarrow f_{\color{blue}{A}}^1 \rightarrow f_{\color{orange}{S}}^1 \rightarrow u_{\color{orange}{S}}^1 \\
u_{\color{blue}{A}}^2 \rightarrow f_{\color{blue}{A}}^2 \rightarrow f_{\color{orange}{S}}^2 \rightarrow u_{\color{orange}{S}}^2
\end{aligned}
```

The residual equations are the following.
```math
\begin{aligned}
{\color{green}D_i}(u_S^{i-1}, u_A^i) = u_A^i - u_A^{i}(u_S^{i-1}) = 0 \newline
{\color{blue}A_i}(u_A^i, f_A^i) = f_A^i - f_A^i(u_A^i) = 0 \\
{\color{green}L_i}(u_S^{i-1},f_A^i, f_S^i) = f_S^i - f_S^i(f_A^i, u_S^{i-1}) = 0 \\
{\color{orange}S_i}(f_S^i, u_S^i) = u_S^i - u_S^i(f_S^i) = 0
\end{aligned}
```

The aeroelastic Lagrangian for some objective function $f(x)$ is the following:
```math
    \mathcal{L}_{AE} = f(x,u_S^i) + \psi_{\color{green}D_i}^T {\color{green}{D}_i} + \psi_{\color{blue}{A}_i}^T {\color{blue}{A}_i} + \psi_{\color{green}{L}_i}^T {\color{green}{L}_i} + \psi_{\color{orange}{S}_i}^T {\color{orange}{S}_i}
```

The individual adjoint equations are:
```math
\begin{aligned}
\frac{\partial \mathcal{L}_{AE}}{\partial u_A^i} = \psi_{\color{green}{D}_i} + \frac{\partial \color{blue}{A}_i}{\partial u_A^i}^T \psi_{\color{blue}{A}_i} = 0 \\
\frac{\partial \mathcal{L}_{AE}}{\partial f_A^i} = \psi_{\color{blue}{A}_i} + \frac{\partial \color{green}{L}_i}{\partial f_A^i}^T \psi_{\color{green}{L}_i} = 0 \\
\frac{\partial \mathcal{L}_{AE}}{\partial f_S^i} = \psi_{\color{green}{L}_i} + \frac{\partial \color{orange}{S}_i}{\partial f_S^i}^T \psi_{\color{orange}{S}_i} = 0 \\
\frac{\partial \mathcal{L}_{AE}}{\partial u_S^i} = \psi_{\color{orange}{S}_i} + \frac{\partial \color{green}{D}_{i+1}}{\partial u_S^i}^T \psi_{\color{green}{D}_{i+1}} + \frac{\partial \color{green}{L}_{i+1}}{\partial u_S^i}^T \psi_{\color{green}{L}_{i+1}} = 0
\end{aligned}
```

The adjoint matrix for a 2-step, aeroelastic unsteady analysis is:
```math
\begin{bmatrix}
1_{\color{blue}{A}} & \frac{\partial \color{blue}{A}_1}{\partial u_A^1}^T & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1_{\color{blue}{A}} & \frac{\partial \color{green}{L}_1}{\partial f_A^1}^T & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1_{\color{orange}{S}} & \frac{\partial \color{orange}{S}_1}{\partial f_S^1}^T & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1_{\color{orange}{S}} & \frac{\partial \color{green}{D}_2}{\partial u_S^1}^T & 0 & \frac{\partial \color{green}{L}_2}{\partial u_S^1}^T & 0 \\
0 & 0 & 0 & 0 & 1_{\color{blue}{A}} & \frac{\partial \color{blue}{A}_2}{\partial u_A^2}^T & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 1_{\color{blue}{A}} & \frac{\partial \color{green}{L}_2}{\partial f_A^2}^T & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1_{\color{orange}{S}} & \frac{\partial \color{orange}{S}_2}{\partial f_S^2}^T \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1_{\color{orange}{S}}
\end{bmatrix}
\begin{bmatrix}
    \psi_{\color{green}{D}_1} \\
    \psi_{\color{blue}{A}_1} \\
    \psi_{\color{green}{L}_1} \\
    \psi_{\color{orange}{S}_1} \\
    \psi_{\color{green}{D}_2} \\
    \psi_{\color{blue}{A}_2} \\
    \psi_{\color{green}{L}_2} \\
    \psi_{\color{orange}{S}_2}
\end{bmatrix} = - \begin{bmatrix}
       \frac{\partial f}{\partial u_A^1}^T \\ \frac{\partial f}{\partial f_A^1}^T \\ 0 \\ \frac{\partial f}{\partial u_S^1} \\ \frac{\partial f}{\partial u_A^2}^T \\ \frac{\partial f}{\partial f_A^2}^T \\ 0 \\ \frac{\partial f}{\partial u_S^2} \end{bmatrix}
```

One important note is that $\psi_{\color{orange}{S}_2} = -\frac{\partial f}{\partial u_S^2}$ which has been fixed in the framework tests.

### Discipline Total Derivatives

$\color{blue}{Aerodynamic}$ design variables:
```math
    \frac{df}{d \color{blue}{x}} = \frac{\partial f}{\partial \color{blue}{x}} + \frac{\partial \color{blue}{A}_i}{\partial \color{blue}{x}}^T \psi_{\color{blue}{A}_i}
```

$\color{orange}{Structural}$ design variables:
```math
    \frac{df}{d \color{orange}{x}} = \frac{\partial f}{\partial \color{orange}{x}} + \frac{\partial \color{orange}{S}_i}{\partial \color{orange}{x}}^T \psi_{\color{orange}{S}_i}
```

### Coordinate Derivatives

$\color{blue}{Aerodynamic}$ coordinate derivatives:
```math
    \frac{df}{d \color{blue}{x_{A0}}} = \frac{\partial \color{green}{D}_i}{\partial \color{blue}{x_{A0}}}^T \psi_{\color{green}{D}_i} + \frac{\partial \color{blue}{A}_i}{\partial \color{blue}{x_{A0}}}^T \psi_{\color{blue}{A}_i} + \frac{\partial \color{green}{D}_i}{\partial \color{blue}{x_{A0}}}^T \psi_{\color{green}{D}_i}
```

$\color{orange}{Structural}$ coordinate derivatives:
```math
    \frac{df}{d \color{orange}{x_{S0}}} = \frac{\partial \color{green}{D}_i}{\partial \color{orange}{x_{S0}}}^T \psi_{\color{green}{D}_i} + \frac{\partial \color{green}{D}_i}{\partial \color{orange}{x_{S0}}}^T \psi_{\color{green}{D}_i} + \frac{\partial \color{orange}{S}_i}{\partial \color{orange}{x_{S0}}}^T \psi_{\color{orange}{S}_i}
```

## Unsteady Aerothermal Analysis
Considering aerothermal adjoint equations to demonstrate unsteady loop, involves four adjoints $\color{green}{T}_k, \color{blue}{A}_k, \color{green}{Q}_k, \color{red}{P}_k$ for each time step $(k)$.

### Forward analysis
Forward analysis state variables loop with $t_S^0 = 0$ initial condition, where $t$ are temperatures and $h$ are heat loads.
```math
\begin{aligned}
    t_{\color{red}{S}}^0 \\
    t_{\color{blue}{A}}^1 \rightarrow h_{\color{blue}{A}}^1 \rightarrow h_{\color{red}{S}}^1 \rightarrow t_{\color{red}{S}}^1 \\
    t_{\color{blue}{A}}^2 \rightarrow h_{\color{blue}{A}}^2 \rightarrow h_{\color{red}{S}}^2 \rightarrow t_{\color{red}{S}}^2 \\
\end{aligned}
```

The residual equations are the following
```math
\begin{aligned}
{\color{green}T_i}(t_S^{i-1}, t_A^i) = t_A^i - t_A^{i}(t_S^{i-1}) = 0 \\
{\color{blue}A_i}(t_A^i, h_A^i) = h_A^i - h_A^i(t_A^i) = 0 \\
{\color{green}Q_i}(h_A^i, h_S^i) = h_S^i - h_S^i(h_A^i) = 0 \\
{\color{red}P_i}(h_S^i, t_S^i) = t_S^i - t_S^i(h_S^i) = 0
\end{aligned}
```

### Adjoint Equations
The Lagrangian for some objective function $f(x)$ is the following:
```math
    \mathcal{L}_{AT} = f(x,u_S^i) + \psi_{\color{green}{T}_i}^T {\color{green}{T}_i} + \psi_{\color{blue}{A}_i}^T {\color{blue}{A}_i} + \psi_{\color{green}{Q}_i}^T {\color{green}{Q}_i} + \psi_{\color{red}{P}_i}^T \color{red}{P}_i
```

The individual adjoint equations are:
```math
\begin{aligned}
% dL/dtAi
    \frac{\partial \mathcal{L}_{AT}}{\partial t_A^i} = \psi_{\color{green}{T}_i} + \frac{\partial \color{blue}{A}_i}{\partial t_A^i}^T \psi_{\color{blue}{A}_i} = 0 \\
% dL/dhAi
    \frac{\partial \mathcal{L}_{AT}}{\partial f_A^i} = \psi_{\color{blue}{A}_i} + \frac{\partial \color{green}{Q}_i}{\partial h_A^i}^T \psi_{\color{green}{Q}_i} = 0 \\
% dL/dhsi
    \frac{\partial \mathcal{L}_{AT}}{\partial h_S^i} = \psi_{\color{green}{L}_i} + \frac{\partial \color{red}{P}_i}{\partial h_S^i}^T \psi_{\color{red}{P}_i} = 0 \\
% dL/dtsi
    \frac{\partial \mathcal{L}_{AT}}{\partial t_S^i} = \psi_{\color{red}{P}_i} + \frac{\partial \color{green}{T}_{i+1}}{\partial t_S^i}^T \psi_{\color{green}{T}_{i+1}} = 0
\end{aligned}
```

The adjoint matrix for a 2-step, aeroelastic unsteady analysis is:
```math
\begin{bmatrix}
        1_{\color{blue}{A}} & \frac{\partial \color{blue}{A}_1}{\partial t_A^1}^T & 0 & 0 & 0 & 0 & 0 & 0 \\
        0 & 1_{\color{blue}{A}} & \frac{\partial \color{green}{Q}_1}{\partial h_A^1}^T & 0 & 0 & 0 & 0 & 0 \\
        0 & 0 & 1_{\color{red}{S}} & \frac{\partial \color{red}{P}_1}{\partial h_S^1}^T & 0 & 0 & 0 & 0 \\
        0 & 0 & 0 & 1_{\color{red}{S}} & \frac{\partial \color{green}{T}_2}{\partial t_S^1}^T & 0 & 0 & 0 \\
        0 & 0 & 0 & 0 & 1_{\color{blue}{A}} & \frac{\partial \color{blue}{A}_2}{\partial t_A^2}^T & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 1_{\color{blue}{A}} & \frac{\partial \color{green}{Q}_2}{\partial h_A^2}^T & 0 \\
        0 & 0 & 0 & 0 & 0 & 0 & 1_{\color{red}{S}} & \frac{\partial \color{red}{P}_2}{\partial h_S^2}^T \\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 1_{\color{red}{S}}
    \end{bmatrix}
    \begin{bmatrix}
        \psi_{\color{green}{T}_1} \\
        \psi_{\color{blue}{A}_1} \\
        \psi_{\color{green}{Q}_1} \\
        \psi_{\color{red}{P}_1} \\
        \psi_{\color{green}{T}_2} \\
        \psi_{\color{blue}{A}_2} \\
        \psi_{\color{green}{Q}_2} \\
        \psi_{\color{red}{P}_2}
    \end{bmatrix} = - \begin{bmatrix}
       \frac{\partial f}{\partial t_A^1}^T \\ \frac{\partial f}{\partial h_A^1}^T \\ 0 \\ \frac{\partial f}{\partial t_S^1} \\ \frac{\partial f}{\partial t_A^2}^T \\ \frac{\partial f}{\partial h_A^2}^T \\ 0 \\ \frac{\partial f}{\partial t_S^2}  \end{bmatrix}
```

### Discipline Total Derivatives

$\color{blue}{Aerodynamic}$ design variables:
```math
    \frac{df}{d \color{blue}{x}} = \frac{\partial f}{\partial \color{blue}{x}} + \frac{\partial \color{blue}{A}_i}{\partial \color{blue}{x}}^T \psi_{\color{blue}{A}_i}
```

$\color{red}{Structural}$ design variables:
```math
    \frac{df}{d \color{red}{x}} = \frac{\partial f}{\partial \color{red}{x}} + \frac{\partial \color{red}{P}_i}{\partial \color{red}{x}}^T \psi_{\color{red}{P}_i}
```

### Coordinate Derivatives

$\color{blue}{Aerodynamic}$ coordinate derivatives:
```math
    \frac{df}{d \color{blue}{x_{A0}}} = \frac{\partial \color{blue}{A}_i}{\partial \color{blue}{x_{A0}}}^T \psi_{\color{blue}{A}_i}
```

$\color{red}{Structural}$ coordinate derivatives:
```math
    \frac{df}{d \color{red}{x_{S0}}} = \frac{\partial \color{red}{P}_i}{\partial \color{red}{x_{S0}}}^T \psi_{\color{red}{P}_i}
```

### Unsteady TACS Discipline Derivatives ###
<figure class="image">
  <img src="images/unsteady_tacs_discipline_tests.drawio.png" width=\linewidth/>
</figure>

### Unsteady Aerodynamic Coordinate Derivatives ###
<figure class="image">
  <img src="images/unsteady_f2f_aero_coords.drawio.png" width=\linewidth/>
</figure>

### Unsteady Structural Coordinate Derivatives ###
<figure class="image">
  <img src="images/unsteady_tacs_struct_coords2.drawio.png" width=\linewidth/>
</figure>
