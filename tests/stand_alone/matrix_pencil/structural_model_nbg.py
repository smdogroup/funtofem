from __future__ import print_function

from pyfuntofem.solver_interface import *
from tacs import TACS, functions
from mpi4py import MPI
import numpy as np
from matrix_pencil import *

class SpringStructure(SolverInterface):
    def __init__(self, comm, model, dtype=np.double,
                 aeroelastic_coupling=True, use_test_function=False):
        """
        See "Fully-Implicit Time-Marching Aeroelastic Solutions" by Juan J.
        Alonso and Antony Jameson

        """
        self.comm = comm
        self.dtype = dtype

        # Set a flag for whether to actually use aeroelastic coupling
        # or not
        self.aeroelastic_coupling = aeroelastic_coupling

        # Set a flag to use a test function in place of the matrix
        # pencil method
        self.use_test_function = use_test_function

        # Define parameters from Alonso paper
        self.b = 0.5 # [m]
        self.a = -2.0
        self.w_h = 100.0
        self.w_alpha = 100.0
        self.w_f = 100.0 # forcing frequency

        # Hard code the non-dimensional parameters
        self.rho_ref = 1.225
        self.x_alpha = 1.8
        self.r_alpha = np.sqrt(3.48)
        self.mu = 60.0

        # Set the forcing period
        self.t_f = 2.0*np.pi/self.w_f

        # Compute dimensional parameters from what's given above
        self.m = self.rho_ref*self.mu*np.pi*self.b**2
        self.I_alpha = self.m*self.b**2*self.r_alpha**2
        self.S_alpha = self.m*self.b*self.x_alpha

        # Compute stiffnesses
        self.k_h = self.m*self.w_h**2
        self.k_alpha = self.I_alpha*self.w_alpha**2

        # Geometric properties of the airfoil grid
        self.p = np.array([(1.0 + self.a)*self.b, 0.0, 0.0])  # m, attachment point
        self.width = 1.0                                      # m, witch of airfoil grid

        # Set the mass and stiffness matrices
        self.K = np.array([[self.k_h, 0.0         ],
                           [0.0,      self.k_alpha]],
                          dtype=self.dtype)
        self.M = np.array([[self.m,       self.S_alpha],
                           [self.S_alpha, self.I_alpha]],
                          dtype=self.dtype)

        # Settings for matrix pencil
        self.N = 250  # the number of points to sample for the matrix pencil
        self.output_levels = 0
        self.pitch_pencil = None
        self.dc = None  # gradient of damping w.r.t. structural states
        self.t_p_fraction = 0.5

        # Set Newmark method
        self.newmark_beta = 0.25
        self.newmark_gamma = 0.5

        if self.comm.Get_rank()==0:
            for body in model.bodies:
                body.struct_nnodes = 4
                body.struct_X = np.array([0.55, 0.0, 0.05,
                                          0.45, 0.0, 0.05,
                                          0.45, 0.0, -0.05,
                                          0.55, 0.0, -0.05], dtype=self.dtype)
                body.struct_disps = np.zeros(12, dtype=self.dtype)
                body.struct_loads = np.zeros(12, dtype=self.dtype)

        else:
            for body in model.bodies:
                body.struct_nnodes = 0
                body.struct_X = np.zeros(0*3, dtype=self.dtype)
                body.struct_disps = np.zeros(0)
                body.struct_loads = np.zeros(0)

        # Set the Mach number (needed to extract out the
        # design-dependence from the structural time step)
        self.minf = None
        self.flow_dt = None
        self.dt = None

        self.alpha0 = 0.0

        # Run the simulation on the root processor alone
        if self.comm.Get_rank() == 0:
            color = 55
            key = self.comm.Get_rank()
        else:
            color = 77
            key = self.comm.Get_rank()
        self.tacs_comm = self.comm.Split(color, key)

        # For debugging purposes
        self.fixed_step = 5
        self.sum_states = 0.0

        return

    def initialize(self, scenario, bodies):
        self.q = [np.zeros(2, dtype=self.dtype)]
        self.qdot = [np.zeros(2, dtype=self.dtype)]
        self.qddot = [np.zeros(2, dtype=self.dtype)]

        # Set the sum-of-states test function to zero
        self.sum_states = 0.0

        # Set the initial alpha
        self.q[0][1] = self.alpha0

        # Reset time history of states to zero between optimization steps
        self.t_hist = np.array([0.0], dtype=self.dtype)
        self.h_hist = np.array([0.0], dtype=self.dtype)
        self.alpha_hist = np.array([self.q[0][1]], dtype=self.dtype)

        # Compute dfdx
        self.dfdx = np.zeros(2, dtype=self.dtype)

        # The adjoint values
        self.psi = np.zeros(2, dtype=self.dtype)
        self.lmbda = np.zeros(2, dtype=self.dtype)
        self.phi = np.zeros(2, dtype=self.dtype)

        # Adjoint variable history
        self.adjoint_hist = []

        self.B = []
        if bodies is not None:
            for body in bodies:
                X = body.struct_X.reshape(-1,3)

                nnodes = X.shape[0]
                r = (X - self.p).flatten()
                B = np.zeros((2, 3*nnodes), dtype=X.dtype)
                B[0,2::3] = -1.0/self.width
                B[1,0::3] = r[2::3]/self.width
                B[1,2::3] = -r[0::3]/self.width

                self.B.append(B)

        return 0

    def iterate_eom(self, step, f):
        '''
        Iterate on the governing equations
        '''

        time = self.dt*step

        if time.real < self.t_f:
            u0 = np.pi/180.0

            # Force the pitch motion with an amplitude of 1 degree
            u = np.zeros(2, dtype=self.dtype)
            u[0] = 0.0
            u[1] = u0*np.sin(self.w_f*time)

            udot = np.zeros(2, dtype=self.dtype)
            udot[0] = 0.0
            udot[1] = u0*np.cos(self.w_f*time)*self.w_f

            uddot = np.zeros(2, dtype=self.dtype)
            uddot[0] = 0.0
            uddot[1] = -u0*np.sin(self.w_f*time)*self.w_f**2
        else:
            # The initial guess for uddot
            uddot = np.array(self.qddot[step-1], dtype=self.dtype)

            # Compute the initial guess for the state variables
            b1 = (0.5 - self.newmark_beta)*self.dt**2
            b2 = self.newmark_beta*self.dt**2
            u = self.q[step-1] + self.dt*self.qdot[step-1] + b1*self.qddot[step-1] + b2*uddot

            # Compute the initial guess for the time derivatives
            g1 = (1.0 - self.newmark_gamma)*self.dt
            g2 = self.newmark_gamma*self.dt
            udot = self.qdot[step-1] + g1*self.qddot[step-1] + g2*uddot

            # Get the residuals of the governing equations
            res = np.dot(self.M, uddot) + np.dot(self.K, u) - f

            # Coefficients on u, udot, and uddot
            alpha = b2
            beta = g2
            gamma = 1.0

            # Compute the Jacobian
            J = alpha*self.K + gamma*self.M

            # Compute the update
            update = np.linalg.solve(J, res)

            # Apply the update
            u -= alpha*update
            udot -= beta*update
            uddot -= gamma*update

        self.q.append(u)
        self.qdot.append(udot)
        self.qddot.append(uddot)

        return u

    def iterate_adjoint_eom(self, step, df):
        time = self.dt*step

        psi = np.zeros(2, dtype=self.dtype)

        if time.real < self.t_f:
            # Compute the beta/gamma coefficients
            b1 = (0.5 - self.newmark_beta)*self.dt**2
            g1 = (1.0 - self.newmark_gamma)*self.dt

            # Take the derivative of the final fixed-time step
            # w.r.t. the derivative of u0 and u0dot w.r.t. dt
            u0 = np.pi/180.0
            dtime = 1.0*step
            du1 = u0*np.cos(self.w_f*time)*dtime*self.w_f
            du1dot = -u0*np.sin(self.w_f*time)*dtime*self.w_f**2
            du1ddot = -u0*np.cos(self.w_f*time)*dtime*self.w_f**3

            # Add the contribution from the initial step before the
            # Newmark-beta method
            self.dfdx[0] += (
                self.lmbda[1]*(-du1 - self.dt*du1dot - b1*du1ddot) +
                self.phi[1]*(-du1dot - g1*du1ddot))

            # Zero out the adjoint terms for the remaining steps
            self.phi[:] = 0.0
            self.lmbda[:] = 0.0
            self.psi[:] = 0.0

            self.dfdx[0] += df[1]*du1
        elif step == 0:
            # Compute the contributions from the derivative w.r.t. q0
            # self.dfdx[1] += df[1] - self.lmbda[1]
            self.dfdx[1] -= self.lmbda[1]
        elif step > 0:
            # Compute the beta/gamma coefficients
            b1 = (0.5 - self.newmark_beta)*self.dt**2
            b2 = self.newmark_beta*self.dt**2
            g1 = (1.0 - self.newmark_gamma)*self.dt
            g2 = self.newmark_gamma*self.dt

            # Solve for the phi value at the next step
            phi = self.phi + self.dt*self.lmbda

            # Set up the right-hand-side for the next step
            J = self.M + b2*self.K

            # Compute the right-hand-side
            rhs = -b2*df + (b1 + b2)*self.lmbda + g2*phi + g1*self.phi

            # Solve for the updated value of phi
            psi = np.linalg.solve(J, rhs)

            # Compute the updated value of lambda
            lmbda = self.lmbda - np.dot(self.K, psi) - df

            # Compute the derivative of the coefficieints
            db1 = 2*self.dt*(0.5 - self.newmark_beta)
            db2 = 2*self.dt*self.newmark_beta
            dg1 = 1.0 - self.newmark_gamma
            dg2 = self.newmark_gamma

            # Add the contributions from the total derivative
            self.dfdx[0] -= (
                np.dot(lmbda, self.qdot[step-1] +
                       (db1*self.qddot[step-1] + db2*self.qddot[step])) +
                np.dot(phi, dg1*self.qddot[step-1] + dg2*self.qddot[step]))

            # Update the adjoint values
            self.psi = psi.copy()
            self.lmbda = lmbda
            self.phi = phi

        # Append to the adjoint history
        self.adjoint_hist.append(psi)

        return psi

    def get_functions(self, scenario, bodies):
        for function in scenario.functions:
            if function.analysis_type == 'structural' and 'displacement' in function.name:
                function.value = self.sum_states
            elif function.analysis_type == 'structural' and 'pitch' in function.name:
                # Evaluate the pitch damping
                c = None

                # Set the offset as a fixed number of steps, rather
                # than as a function of time
                offset = int(self.t_p_fraction*scenario.steps)
                if self.comm.Get_rank() == 0:
                    if self.use_test_function:
                        c = 0.5*np.sum(self.alpha_hist**2)
                    else:
                        # HACK: Make the time-step independent of the
                        # design variable vf
                        vinf = self.flow_dt*self.minf/self.dt
                        vf = vinf/(self.b*self.w_alpha*np.sqrt(self.mu))

                        # Cut forcing period out of signal and scale by vf
                        # to eliminate design dependence of T on vf
                        T = vf*self.t_hist[offset:]
                        U = self.alpha_hist[offset:]

                        # Apply matrix pencil method to get the objective function
                        if len(U) < self.N:
                            self.pitch_pencil = MatrixPencil(T, U, -1, self.output_levels)
                        else:
                            self.pitch_pencil = MatrixPencil(T, U, self.N, self.output_levels)

                        self.pitch_pencil.ComputeDampingAndFrequency()
                        c = self.pitch_pencil.AggregateDamping()

                    print("Aggregate damping, c = ", c)

                c = self.comm.bcast(c)

                function.value = self.comm.bcast(c)

    def set_variables(self, scenario, bodies):
        # Set design variables
        if 'structural' in scenario.variables:
            for var in scenario.variables['structural']:
                if 'struct_dt' == var.name.lower():
                    self.dt = var.value
                elif 'alpha0' == var.name.lower():
                    self.alpha0 = var.value

        return

    def set_states(self, scenario, bodies, step):
        h = None
        alpha = None

        if self.comm.Get_rank() == 0:
            h = self.h_hist[step]
            alpha = self.alpha_hist[step]

        h = self.comm.bcast(h)
        alpha = self.comm.bcast(alpha)

        X = bodies[0].struct_X.reshape((-1,3))
        U = convert_disps(h, alpha, self.p, X)

        bodies[0].struct_disps = U.flatten()

        return 0

    def iterate(self, scenario, bodies, step):
        if self.comm.Get_rank() == 0:
            for ibody, body in enumerate(bodies):
                # Set the loads
                f = np.zeros(2, dtype=self.dtype)

                if self.aeroelastic_coupling:
                    # Compute the sectional lift and moment forces
                    X = body.struct_X.reshape(-1,3)

                    # Compute the forces based on the structural loads
                    f = np.dot(self.B[ibody], body.struct_loads)

                    if step == self.fixed_step:
                        self.sum_states = np.sum(f)

                # Take a step
                u = self.iterate_eom(step, f)

                # Extract the plunge and pitch values
                h = u[0]
                alpha = u[1]

                # Record time and states in histories
                self.t_hist = np.append(self.t_hist, self.dt*step)
                self.h_hist = np.append(self.h_hist, h)
                self.alpha_hist = np.append(self.alpha_hist, alpha)

                if self.aeroelastic_coupling:
                    # Convert states to displacements of surface mesh
                    U = convert_disps(h, alpha, self.p, X)
                    body.struct_disps = U.flatten()
        else:
            for body in bodies:
                body.struct_disps = np.zeros(body.struct_nnodes*3,
                                             dtype=TACS.dtype)

        return 0

    def post(self, scenario, bodies):
        if self.comm.Get_rank() == 0:
            f = open('time_hist.dat', 'w')
            for i in range(len(self.t_hist)):
                f.write('{0} {1} {2}\n'.format(self.t_hist[i].real,
                                               self.h_hist[i].real,
                                               self.alpha_hist[i].real))

            f.close()

        return

    def post_adjoint(self, scenario, bodies):
        if self.comm.rank == 0:
            with open('adjoint_hist.dat', 'w') as fp:
                hist = self.adjoint_hist[::-1]
                for i in range(len(hist)):
                    fp.write('{0} {1} {2}\n'.format((i*self.dt).real,
                                                    hist[i][0].real,
                                                    hist[i][1].real))


    def initialize_adjoint(self, scenario, bodies):
        # Specify the number of design variables and the functions to the
        # integrator (use Structural Mass as a dummy function)

        # Compute the right-hand side for the structural adjoint
        dc = None

        for function in scenario.functions:
            if function.analysis_type == 'structural' and 'pitch' in function.name:
                if self.comm.Get_rank() == 0:
                    if self.use_test_function:
                        dc = np.array(self.alpha_hist, dtype=self.dtype)
                    else:
                        dcdx = self.pitch_pencil.AggregateDampingDer()
                        npad = len(self.t_hist) - len(dcdx)
                        dc = np.hstack((np.zeros(npad), dcdx))

        self.dc = self.comm.bcast(dc)

        return 0

    def iterate_adjoint(self, scenario, bodies, step):
        if self.comm.Get_rank() == 0:
            for ibody, body in enumerate(bodies):
                # Create the vector for the right-hand-side of the
                # adjoint equations
                df = np.zeros(2, dtype=self.dtype)
                if self.dc is not None:
                    df[1] = self.dc[step]

                X = body.struct_X.reshape(-1,3)
                if self.aeroelastic_coupling:
                    # Get the plunge/alpha values for this step
                    h = self.h_hist[step]
                    alpha = self.alpha_hist[step]

                    # Solve displacement conversion adjoint equation
                    psi_Y = body.struct_rhs[:,0].reshape((-1,3))

                    # Perform the transformation to the plunge/pitch
                    # adjoint right-hand-sides
                    psi_h, psi_alpha = convert_disps_transpose(h, alpha, self.p, X,
                                                               psi_Y)

                    df[0] -= psi_h
                    df[1] -= psi_alpha

                # Solve structural adjoint equation
                psi_S = self.iterate_adjoint_eom(step, df)

                if step == self.fixed_step:
                    for ifunc, func in enumerate(scenario.functions):
                        if func.analysis_type == 'structural' and 'displacement' in func.name:
                            psi_S[:] -= 1.0

                if self.aeroelastic_coupling:
                    # Solve load transfer adjoint equation
                    body.psi_S[:,0] = np.dot(self.B[ibody].T, psi_S)

        return 0

    def get_function_gradients(self, scenario, bodies, offset):
        for ifunc, func in enumerate(scenario.functions):
            # Place the gradient in the correct location
            for vartype in scenario.variables:
                if vartype == 'structural':
                    for i, var in enumerate(scenario.variables[vartype]):
                        if var.name == 'alpha0' and var.active:
                            if self.comm.rank == 0:
                                scenario.derivatives[vartype][offset+ifunc][i] = self.dfdx[1]

                            scenario.derivatives[vartype][offset+ifunc][i] =\
                                self.comm.bcast(scenario.derivatives[vartype][offset+ifunc][i],root=0)

                        if var.name == 'struct_dt' and var.active:
                            if self.comm.rank == 0:
                                scenario.derivatives[vartype][offset+ifunc][i] = self.dfdx[0]

                            scenario.derivatives[vartype][offset+ifunc][i] =\
                                self.comm.bcast(scenario.derivatives[vartype][offset+ifunc][i],root=0)

    def eval_func(self, x, name='dt'):
        '''
        For testing purposes only:

        Evaluate the sum of squares of the alpha history using dt = x
        as the design variable. This only tests the equations of
        motion not the matrix pencil part.
        '''
        if name == 'dt':
            self.dt = x
        elif name == 'alpha0':
            self.alpha0 = x
        self.initialize(None, None)

        f = np.zeros(2, dtype=self.dtype)
        steps = 2000

        self.alpha_hist = [0.0]
        for i in range(1, steps):
            u = self.iterate_eom(i, f)
            self.alpha_hist.append(u[1])

        self.alpha_hist = np.array(self.alpha_hist)
        with open('alpha_hist.dat', 'w') as fp:
            for i in range(len(self.alpha_hist)):
                fp.write('{0} {1}\n'.format(i*self.dt, self.alpha_hist[i]))

        return 0.5*np.sum(self.alpha_hist**2)

    def eval_grad(self):
        '''
        Evaluate the gradient of the sum of the squares of the alpha
        history
        '''
        df = np.zeros(2, dtype=self.dtype)

        steps = 2000
        for i in range(steps-1, -1, -1):
            df[1] = self.alpha_hist[i]
            psi = self.iterate_adjoint_eom(i, df)

        with open('adjoint_hist.dat', 'w') as fp:
            hist = self.adjoint_hist[::-1]
            for i in range(len(hist)):
                fp.write('{0} {1} {2}\n'.format(i*self.dt, hist[i][0],
                                                hist[i][1]))

        return self.dfdx

def convert_disps(h, alpha, p, X):
    """
    Convert states to airfoil coordinate displacements

    Parameters
    ----------
    h : float
        plunge coordinate
    alpha : float
        pitch coordinate
    p : numpy ndarray
        location of attachment point
    X : numpy ndarray
        airfoil coordinates

    Returns
    -------
    U : numpy ndarray
        airfoil displacements

    """
    # Form rigid rotation and translation
    R = np.zeros((3,3), dtype=X.dtype)
    R[0,0] = np.cos(alpha)
    R[2,0] = -np.sin(alpha)
    R[1,1] = 1.0
    R[0,2] = np.sin(alpha)
    R[2,2] = np.cos(alpha)

    t = np.zeros(3, dtype=X.dtype)
    t[2] = -h

    # Form displacements
    r = X - p
    U = r.dot(R.T) + p + t - X

    return U

def convert_disps_transpose(h, alpha, p, X, psi):
    """
    Action of displacement conversion Jacobian on adjoint vector

    Parameters
    ----------
    alpha : float
        pitch coordinate
    p : numpy ndarray
        location of attachment point
    X : numpy ndarray
        airfoil coordinates
    Vec : numpy ndarray
        adjoint vector

    Returns
    -------
    prod : numpy ndarray
        structural adjoint variables

    """
    dRdalpha = np.zeros((3,3), dtype=X.dtype)
    dRdalpha[0,0] = -np.sin(alpha)
    dRdalpha[2,0] = -np.cos(alpha)
    dRdalpha[0,2] = np.cos(alpha)
    dRdalpha[2,2] = -np.sin(alpha)

    r = X - p

    psi_h = -np.sum(psi[:,2])
    psi_alpha = np.sum(psi*r.dot(dRdalpha.T))

    return psi_h, psi_alpha

def test_transfer_transpose():
    n = 20
    X = -1.0 + 2.0*np.random.uniform(size=(n, 3))
    psi = np.random.uniform(size=(n, 3))

    alpha = -0.5 + 1.0*np.random.uniform()
    h = -0.25 + 0.5*np.random.uniform()
    p = -2.0

    psi_h, psi_alpha = convert_disps_transpose(h, alpha, p, X, psi)

    for dh in [1e-6, 5e-7, 1e-7]:
        fd_h = 0.5*(np.sum(psi*convert_disps(h + dh, alpha, p, X)) -
                    np.sum(psi*convert_disps(h - dh, alpha, p, X)))/dh
        print('psi_h:     %25.15e %25.15e %25.15e'%(fd_h, psi_h, (fd_h - psi_h)/fd_h))

        fd_alpha = 0.5*(np.sum(psi*convert_disps(h, alpha + dh, p, X)) -
                        np.sum(psi*convert_disps(h, alpha - dh, p, X)))/dh
        print('psi_alpha: %25.15e %25.15e %25.15e'%(fd_alpha, psi_alpha,
                                                    (fd_alpha - psi_alpha)/fd_alpha))


def test_adjoint_eom():
    # Test the
    from mpi4py import MPI
    from pyfuntofem.model import FUNtoFEMmodel

    model = FUNtoFEMmodel('spring-mounted airfoil')

    comm = MPI.COMM_WORLD
    sp = SpringStructure(comm, model, dtype=np.complex)

    dt = 5e-4
    fval = sp.eval_func(dt, name='dt')
    grad = sp.eval_grad()

    print('%25s %25s %25s'%('gradient', 'cs', 'rel. error'))
    for exponent in np.linspace(-10, -15, 10):
        dh = 10**exponent
        fd = sp.eval_func(dt + 1j*dh, name='dt').imag/dh
        print('%25.10e %25.10e %25.10e'%(grad[0].real, fd,
                                         (grad[0].real - fd)/fd))

    sp.dt = dt

    sp.t_f = -1.0
    alpha0 = 0.2
    fval = sp.eval_func(alpha0, name='alpha0')
    grad = sp.eval_grad()

    print('%25s %25s %25s'%('gradient', 'cs', 'rel. error'))
    for exponent in np.linspace(-10, -15, 10):
        dh = 10**exponent
        fd = sp.eval_func(alpha0 + 1j*dh, name='alpha0').imag/dh
        print('%25.10e %25.10e %25.10e'%(grad[1].real, fd,
                                         (grad[1].real - fd)/fd))


if __name__ == '__main__':
    test_transfer_transpose()
    test_adjoint_eom()
