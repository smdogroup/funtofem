# Unsteady Framework Tests #
* `test_framework_adjoint_eqns.py` - hardcode the unsteady adjoint equation matrix and total derivatives for two unsteady steps, to verify the unsteady framework.
* `test_framework_unsteady.py` - Test the fully-coupled unsteady analysis with the TestAero+TestStruct solvers.
* `test_funtofem_unsteady_aero_coord.py` - Test the aerodynamic coordinate derivatives of a fully-coupled unsteady analysis with the TestAero + TACS solvers.
* `test_funtofem_unsteady_struct_coord.py` - Test the structural coordinate derivatives of a fully-coupled unsteady analysis with the TestAero + TACS solvers.
* `test_tacs_adjoint_equations.py` - this test is still in dev, but essentially hardcodes the adojint equations from the unsteady adjoint matrix. (since the integration tests work though, it is probably not needed anymore).
* `test_tacs_driver_unsteady_coordinate.py` - Test the structural coordinate derivatives of a oneway-coupled TACS analysis.
* `test_tacs_interface_unsteady.py` - Test the structural and aerodynamic derivatives of a fully-coupled unsteady analysis with TestAero + TACS solvers.
* `test_tacs_unsteady_shape_driver.py` - Test the structural shape derivatives in ESP/CAPS using the TACS AIM in an unsteady, fully-coupled analysis with TestAero + TACS solvers.
* `test_unsteady_solvers.py` - directional derivative test for the two test aero + test struct solvers for unsteady.

## Unsteady Aeroelastic Analysis ##
The unsteady aeroelastic forward analysis involves displacements $u_{\blue{A}}^i, u_{\orange{S}}^i$ and forces $f_{\blue{A}}^i, f_{\orange{S}}^i$ for each time step $i$.
$$
u_{\orange{S}}^0 \\
u_{\blue{A}}^1 \rightarrow f_{\blue{A}}^1 \rightarrow f_{\orange{S}}^1 \rightarrow u_{\orange{S}}^1 \\
u_{\blue{A}}^2 \rightarrow f_{\blue{A}}^2 \rightarrow f_{\orange{S}}^2 \rightarrow u_{\orange{S}}^2
$$

The residual equations are the following.
$$
\green{D}_i(u_S^{i-1}, u_A^i) = u_A^i - u_A^{i}(u_S^{i-1}) = 0 \\
\blue{A}_i(u_A^i, f_A^i) = f_A^i - f_A^i(u_A^i) = 0 \\
\green{L}_i(u_S^{i-1},f_A^i, f_S^i) = f_S^i - f_S^i(f_A^i, u_S^{i-1}) = 0 \\
\orange{S}_i(f_S^i, u_S^i) = u_S^i - u_S^i(f_S^i) = 0
$$

The aeroelastic Lagrangian for some objective function $f(x)$ is the following:
$$
    \mathcal{L}_{AE} = f(x,u_S^i) + \psi_{\green{D}_i}^T \green{D}_i + \psi_{\blue{A}_i}^T \blue{A}_i + \psi_{\green{L}_i}^T \green{L}_i + \psi_{\orange{S}_i}^T \orange{S}_i
$$

The individual adjoint equations are:
$$
\frac{\partial \mathcal{L}_{AE}}{\partial u_A^i} = \psi_{\green{D}_i} + \frac{\partial \blue{A}_i}{\partial u_A^i}^T \psi_{\blue{A}_i} = 0 \\
\frac{\partial \mathcal{L}_{AE}}{\partial f_A^i} = \psi_{\blue{A}_i} + \frac{\partial \green{L}_i}{\partial f_A^i}^T \psi_{\green{L}_i} = 0 \\
\frac{\partial \mathcal{L}_{AE}}{\partial f_S^i} = \psi_{\green{L}_i} + \frac{\partial \orange{S}_i}{\partial f_S^i}^T \psi_{\orange{S}_i} = 0 \\
\frac{\partial \mathcal{L}_{AE}}{\partial u_S^i} = \psi_{\orange{S}_i} + \frac{\partial \green{D}_{i+1}}{\partial u_S^i}^T \psi_{\green{D}_{i+1}} + \frac{\partial \green{L}_{i+1}}{\partial u_S^i}^T \psi_{\green{L}_{i+1}]} = 0
$$

The adjoint matrix for a 2-step, aeroelastic unsteady analysis is:
$$
\begin{bmatrix}
1_{\blue{A}} & \frac{\partial \blue{A}_1}{\partial u_A^1}^T & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1_{\blue{A}} & \frac{\partial \green{L}_1}{\partial f_A^1}^T & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1_{\orange{S}} & \frac{\partial \orange{S}_1}{\partial f_S^1}^T & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1_{\orange{S}} & \frac{\partial \green{D}_2}{\partial u_S^1}^T & 0 & \frac{\partial \green{L}_2}{\partial u_S^1}^T & 0 \\
0 & 0 & 0 & 0 & 1_{\blue{A}} & \frac{\partial \blue{A}_2}{\partial u_A^2}^T & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 1_{\blue{A}} & \frac{\partial \green{L}_2}{\partial f_A^2}^T & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1_{\orange{S}} & \frac{\partial \orange{S}_2}{\partial f_S^2}^T \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1_{\orange{S}}
\end{bmatrix}
\begin{bmatrix}
    \psi_{\green{D}_1} \\
    \psi_{\blue{A}_1} \\
    \psi_{\green{L}_1} \\
    \psi_{\orange{S}_1} \\
    \psi_{\green{D}_2} \\
    \psi_{\blue{A}_2} \\
    \psi_{\green{L}_2} \\
    \psi_{\orange{S}_2}
\end{bmatrix} = - \begin{bmatrix}
       \frac{\partial f}{\partial u_A^1}^T \\ \frac{\partial f}{\partial f_A^1}^T \\ 0 \\ \frac{\partial f}{\partial u_S^1} \\ \frac{\partial f}{\partial u_A^2}^T \\ \frac{\partial f}{\partial f_A^2}^T \\ 0 \\ \frac{\partial f}{\partial u_S^2} \end{bmatrix}
$$

One important note is that $\psi_{\orange{S}_2} = -\frac{\partial f}{\partial u_S^2}$ which has been fixed in the framework tests.

\subsection{Discipline Total Derivatives}
$\blue{Aerodynamic}$ design variables:
$$
    \frac{df}{d \blue{x}} = \frac{\partial f}{\partial \blue{x}} + \frac{\partial \blue{A}_i}{\partial \blue{x}}^T \psi_{\blue{A}_i}
$$
$\orange{Structural}$ design variables:
$$
    \frac{df}{d \orange{x}} = \frac{\partial f}{\partial \orange{x}} + \frac{\partial \orange{S}_i}{\partial \orange{x}}^T \psi_{\orange{S}_i}
$$

\subsection{Coordinate Derivatives}
$\blue{Aerodynamic}$ coordinate derivatives:
$$
    \frac{df}{d \blue{x_{A0}}} = \frac{\partial \green{D}_i}{\partial \blue{x_{A0}}}^T \psi_{\green{D}_i} + \frac{\partial \blue{A}_i}{\partial \blue{x_{A0}}}^T \psi_{\blue{A}_i} + \frac{\partial \green{D}_i}{\partial \blue{x_{A0}}}^T \psi_{\green{D}_i}
$$
$\orange{Structural}$ coordinate derivatives:
$$
    \frac{df}{d \orange{x_{S0}}} = \frac{\partial \green{D}_i}{\partial \orange{x_{S0}}}^T \psi_{\green{D}_i} + \frac{\partial \green{D}_i}{\partial \orange{x_{S0}}}^T \psi_{\green{D}_i} + \frac{\partial \orange{S}_i}{\partial \orange{x_{S0}}}^T \psi_{\orange{S}_i}
$$

### Unsteady Aerothermal Analysis ###
Considering aerothermal adjoint equations to demonstrate unsteady loop, involves four adjoints $\green{T}_k, \blue{A}_k, \green{Q}_k, \red{P}_k$ for each time step $(k)$.

\subsection{Forward analysis}
Forward analysis state variables loop with $t_S^0 = 0$ initial condition, where $t$ are temperatures and $h$ are heat loads.
$$
    t_{\red{S}}^0 \\
    t_{\blue{A}}^1 \rightarrow h_{\blue{A}}^1 \rightarrow h_{\red{S}}^1 \rightarrow t_{\red{S}}^1 \\
    t_{\blue{A}}^2 \rightarrow h_{\blue{A}}^2 \rightarrow h_{\red{S}}^2 \rightarrow t_{\red{S}}^2 \\
$$

The residual equations are the following
$$
\green{T}_i(t_S^{i-1}, t_A^i) = t_A^i - t_A^{i}(t_S^{i-1}) = 0 \\
\blue{A}_i(t_A^i, h_A^i) = h_A^i - h_A^i(t_A^i) = 0 \\
\green{Q}_i(h_A^i, h_S^i) = h_S^i - h_S^i(h_A^i) = 0 \\
\red{P}_i(h_S^i, t_S^i) = t_S^i - t_S^i(h_S^i) = 0
$$

\subsection{Adjoint Equations}
The Lagrangian for some objective function $f(x)$ is the following:
$$
    \mathcal{L}_{AT} = f(x,u_S^i) + \psi_{\green{T}_i}^T \green{T}_i + \psi_{\blue{A}_i}^T \blue{A}_i + \psi_{\green{Q}_i}^T \green{Q}_i + \psi_{\red{P}_i}^T \red{P}_i
$$

The individual adjoint equations are:
$$
% dL/dtAi
    \frac{\partial \mathcal{L}_{AT}}{\partial t_A^i} = \psi_{\green{T}_i} + \frac{\partial \blue{A}_i}{\partial t_A^i}^T \psi_{\blue{A}_i} = 0 \\
% dL/dhAi
    \frac{\partial \mathcal{L}_{AT}}{\partial f_A^i} = \psi_{\blue{A}_i} + \frac{\partial \green{Q}_i}{\partial h_A^i}^T \psi_{\green{Q}_i} = 0 \\
% dL/dhsi
    \frac{\partial \mathcal{L}_{AT}}{\partial h_S^i} = \psi_{\green{L}_i} + \frac{\partial \red{P}_i}{\partial h_S^i}^T \psi_{\red{P}_i} = 0 \\
% dL/dtsi
    \frac{\partial \mathcal{L}_{AT}}{\partial t_S^i} = \psi_{\red{P}_i} + \frac{\partial \green{T}_{i+1}}{\partial t_S^i}^T \psi_{\green{T}_{i+1}} = 0
$$

The adjoint matrix for a 2-step, aeroelastic unsteady analysis is:
$$
\begin{bmatrix}
        1_{\blue{A}} & \frac{\partial \blue{A}_1}{\partial t_A^1}^T & 0 & 0 & 0 & 0 & 0 & 0 \\
        0 & 1_{\blue{A}} & \frac{\partial \green{Q}_1}{\partial h_A^1}^T & 0 & 0 & 0 & 0 & 0 \\
        0 & 0 & 1_{\red{S}} & \frac{\partial \red{P}_1}{\partial h_S^1}^T & 0 & 0 & 0 & 0 \\
        0 & 0 & 0 & 1_{\red{S}} & \frac{\partial \green{T}_2}{\partial t_S^1}^T & 0 & 0 & 0 \\
        0 & 0 & 0 & 0 & 1_{\blue{A}} & \frac{\partial \blue{A}_2}{\partial t_A^2}^T & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 1_{\blue{A}} & \frac{\partial \green{Q}_2}{\partial h_A^2}^T & 0 \\
        0 & 0 & 0 & 0 & 0 & 0 & 1_{\red{S}} & \frac{\partial \red{P}_2}{\partial h_S^2}^T \\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 1_{\red{S}}
    \end{bmatrix}
    \begin{bmatrix}
        \psi_{\green{T}_1} \\
        \psi_{\blue{A}_1} \\
        \psi_{\green{Q}_1} \\
        \psi_{\red{P}_1} \\
        \psi_{\green{T}_2} \\
        \psi_{\blue{A}_2} \\
        \psi_{\green{Q}_2} \\
        \psi_{\red{P}_2}
    \end{bmatrix} = - \begin{bmatrix}
       \frac{\partial f}{\partial t_A^1}^T \\ \frac{\partial f}{\partial h_A^1}^T \\ 0 \\ \frac{\partial f}{\partial t_S^1} \\ \frac{\partial f}{\partial t_A^2}^T \\ \frac{\partial f}{\partial h_A^2}^T \\ 0 \\ \frac{\partial f}{\partial t_S^2}  \end{bmatrix}
$$

\subsection{Discipline Total Derivatives}
$\blue{Aerodynamic}$ design variables:
$$
    \frac{df}{d \blue{x}} = \frac{\partial f}{\partial \blue{x}} + \frac{\partial \blue{A}_i}{\partial \blue{x}}^T \psi_{\blue{A}_i}
$$
$\red{Structural}$ design variables:
$$
    \frac{df}{d \red{x}} = \frac{\partial f}{\partial \red{x}} + \frac{\partial \red{P}_i}{\partial \red{x}}^T \psi_{\red{P}_i}
$$

\subsection{Coordinate Derivatives}
$\blue{Aerodynamic}$ coordinate derivatives:
$$
    \frac{df}{d \blue{x_{A0}}} = \frac{\partial \blue{A}_i}{\partial \blue{x_{A0}}}^T \psi_{\blue{A}_i}
$$
$\red{Structural}$ coordinate derivatives:
$$
    \frac{df}{d \red{x_{S0}}} = \frac{\partial \red{P}_i}{\partial \red{x_{S0}}}^T \psi_{\red{P}_i}
$$

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