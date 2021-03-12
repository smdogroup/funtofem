from __future__ import print_function

import numpy as np
import scipy.linalg as la
import scipy.signal as sig
import bisect
import matplotlib.pyplot as plt
from matrix_pencil_der import *

class MatrixPencil(object):
    def __init__(self, t, x, N=-1, output_level=0, rho=100):
        """
        Provide the capability to:
        1) decompose input signal into a series of complex exponentials by the
        Matrix Pencil Method; and
        2) compute the derivative of the KS (Kreisselmeier-Steinhauser)
        aggregation of the real exponents (damping) with respect to the input
        data.

        Parameters
        ----------
        t : numpy.ndarray
            time array (must be evenly spaced)
        x : numpy.ndarray
            array containing sampled waveform
        N : int
            number of samples to downsample to (optional)
        output_level : bool
            choose different output levels (optional)
        rho : float
            KS parameter (default=100)

        """
        self.output_level = "{0:02b}".format(output_level)
        self.rho = rho
        self.t0 = t[0]

        if self.output_level == 1:
            print("Intializing matrix pencil method...")

        # If the samples are complex, separate the imaginary parts
        self.is_complex = False
        if x.dtype == np.complex128 or x.dtype == complex:
            self.is_complex = True
            self.x_imag = x.imag
            x = x.real

        # Downsample the data and save the linear interpolation matrix
        if N == -1:
            self.X = x
            self.N = self.X.shape[0]
            self.dt = t[1] - t[0]
            self.H = np.eye(self.N)

            # check that the step size is uniform
            flag = 0
            tol = 1e-6
            for i in range(t.size-1):
                if abs(self.dt - (t[i+1] - t[i])) > tol:
                    flag = 1
            if flag > 0:
                print("Warning: Matrix pencil time step size is not uniform")
                print("         Please provide data with a uniform step size or use the downsampling option")

        else:
            T = np.linspace(t[0], t[-1], N)
            self.H = self.GenerateLinearInterpMat(T, t, x)
            self.X = self.H.dot(x)
            self.N = self.X.shape[0]
            self.dt = T[1] - T[0]

        # Set the pencil parameter L
        self.n = self.X.shape[0]
        self.L = self.N/2 - 1

        if self.output_level[-1] == "1":
            print("Number of samples, N = ", self.N)
            print("Pencil parameter, L = ", self.L)

        # Save model order
        self.M = None

        # Save SVD of Hankel matrix
        self.U = None
        self.s = None
        self.VT = None

        # Save the filtered right singular vectors V1 and the pseudoinverse
        self.V1T = None
        self.V1inv = None

        # Save right singular vectors V2
        self.V2T = None

        # Save eigenvalues and eigenvectors of A matrix
        self.lam = None
        self.W = None
        self.V = None

        # Save amplitudes, damping, frequencies, and phases
        self.amps = None
        self.damp = None
        self.freq = None
        self.faze = None

    def GenerateLinearInterpMat(self, T, t, x):
        """
        Generate matrix that linearly interpolates the data (t, x) and returns X
        corresponding to T

        """
        N = T.shape[0]
        n = t.shape[0]

        H = np.zeros((N, n))

        if self.output_level[-1] == "1":
            print("Downsampling from {0} to {1} points...".format(n, N))

        for i in range(N):
            # Evaluate first and last basis functions
            j1 = bisect.bisect(t, T[i])

            # Do not allow the first or last points to be selected
            if j1 == 0:
                j1 = 1
                j0 = 0
            elif j1 == n:
                j1 = n-1
                j0 = n-2
            else:
                j0 = j1-1

            dt = t[j1] - t[j0]
            H[i,j0] += (t[j1] - T[i])/dt
            H[i,j1] += (T[i] - t[j0])/dt

        return H

    def ComputeDampingAndFrequency(self):
        """
        Compute the damping and frequencies of the complex exponentials

        """
        # Assemble the Hankel matrix Y from the samples X
        Y = np.empty((self.N-self.L, self.L+1), dtype=self.X.dtype)
        for i in range(self.N-self.L):
            for j in range(self.L+1):
                Y[i,j] = self.X[i+j]

        # Compute the SVD of the Hankel matrix
        self.U, self.s, self.VT = la.svd(Y)

        # Estimate the modal order M based on the singular values
        self.EstimateModelOrder()

        # Filter the right singular vectors of the Hankel matrix based on M
        Vhat = self.VT[:self.M,:]
        self.V1T = Vhat[:,:-1]
        self.V2T = Vhat[:,1:]

        # Compute the pseudoinverse of V1T
        self.V1inv = la.pinv(self.V1T)

        # Form A matrix from V1T and V2T to reduce to eigenvalue problem
        # Jan's method -- recorded in the original Aviation paper
        # A = self.V1inv.dot(self.V2T)

        # GJK method - should only be an M x M system. No eigenvalues
        # of very small value.
        A = self.V2T.dot(self.V1inv)

        # Solve eigenvalue problem to obtain poles
        self.lam, self.W, self.V = la.eig(A, left=True, right=True)

        # Compute damping and freqency
        s = np.log(self.lam[:self.M])/self.dt
        self.damp = s.real
        self.freq = s.imag

        # force the damping of zero frequency mode to something that
        # won't affect the KS aggregation
        for i in range(self.damp.size):
            if abs(self.freq[i]) < 1e-7:
                self.damp[i] = -99999.0

        return

    def EstimateModelOrder(self):
        """
        Store estimate of model order based on input singular values

        Notes
        -----
        This is a pretty crude method for estimating the model order.
        Development of a more sophisticated method would probably yield greater
        robustness

        """
        # Normalize the singular values by the maximum and cut out modes
        # corresponding to singular values below a specified tolerance
        tol1 = 1.0e-2
        snorm = self.s/self.s.max()
        n_above_tol = len(self.s[snorm > tol1])

        # Approximate second derivative singular values using convolve as a
        # central difference operator
        w = [1.0, -1.0]
        diff = sig.convolve(snorm, w, 'valid')
        diffdiff = sig.convolve(diff, w, 'valid')

        # Cut out more modes depending on the approximated second derivative
        # The idea is sort of to cut at an inflection point in the singular
        # value curve or maybe where they start to bottom out
        tol2 = 1.0e-3
        n_bottom_out = 2 + len(diffdiff[diffdiff > tol2])

        # Estimate the number of modes (model order) to have at least two but
        # otherwise informed by the cuts made above
        self.M = min(max(2, min(n_above_tol, n_bottom_out)), self.L)

        # Report the model order
        if self.output_level[-1] == "1":
            print("Model order, M = ", self.M)
            np.savetxt('singular_values.dat',snorm)

        # Plotting to help diagnose what the cuts are doing
        if self.output_level[-2] == "1":
            # Plot normalized singular values and first cut-off
            plt.figure(figsize=(8, 10))
            ax = plt.gca()
            ax.scatter(np.arange(len(self.s)), snorm, s=40, c='blue', alpha=0.3)
            ax.plot(np.array([-0.2*len(self.s), 1.2*len(self.s)]),
                    tol1*np.ones(2),
                    color='orange', linestyle='dashed', linewidth=2,
                    label='cut-off')
            ax.set_yscale('log')
            ax.set_ylim(1e-20, 1e1)
            plt.legend()
            plt.title('Normalized Singular Values', fontsize=16)

            # Plot approximate second derivative and second cut-off
            plt.figure(figsize=(8, 10))
            ax = plt.gca()
            ax.scatter(np.arange(len(diffdiff)), diffdiff,
                       s=40, c='blue', alpha=0.3)
            ax.plot(np.array([-0.2*len(self.s), 1.2*len(self.s)]),
                    tol2*np.ones(2),
                    color='orange', linestyle='dashed', linewidth=2,
                    label='2nd cut-off')
            ax.set_yscale('log')
            ax.set_ylim(1e-20, 1e1)
            plt.legend()
            plt.title('Approx. 2nd Derivative of Singular Values', fontsize=16)
            plt.show()

        return

    def AggregateDamping(self):
        """
        Kreisselmeier-Steinhauser (KS) function to approximate maximum real
        part of exponent

        Returns
        -------
        float
            approximate maximum of real part of exponents

        """

        if self.output_level[-1] == "1":
            print("M = ", self.M)
            print("damping modes are:")
            for i in range(self.damp.size):
                print('freq:', self.freq[i], 'damp:', self.damp[i])

        m = self.damp.max()
        c = m + np.log(np.sum(np.exp(self.rho*(self.damp - m))))/self.rho

        if self.is_complex:
            dcdx = self.AggregateDampingDer()
            return  c + 1j*dcdx.dot(self.x_imag)
        else:
            return  c

    def AggregateDampingDer(self):
        """
        Use the data saved prior to computing the aggregate damping to compute
        the derivative of the damping with respect to the input data

        """
        dcda = DcDalpha(self.damp, self.rho)

        # zero out the contribution from the zero frequency mode if present
        for i in range(self.freq.size):
            if abs(self.freq[i]) < 1e-7:
                dcda[i] = 0.0
        dcdl = DalphaDlamTrans(dcda, self.lam, self.dt)
        dcdA = DlamDATrans(dcdl, self.W, self.V)
        dcdV1T = dAdV1Trans(dcdA, self.V1T, self.V1inv, self.V2T)
        dcdV2T = dAdV2Trans(dcdA, self.V1inv)
        dcdVhat = dV12dVhatTrans(dcdV1T, dcdV2T)
        dcdY = dVhatdYTrans(dcdVhat, self.U, self.s, self.VT)
        dcdX = dYdXTrans(dcdY)
        dcdx = self.H.T.dot(dcdX)

        return  dcdx

    def ComputeAmplitudeAndPhase(self):
        """
        Compute the amplitudes and phases of the complex exponentials

        """
        # Compute the residues
        B = np.zeros((self.N, self.M), dtype=self.lam.dtype)
        for i in range(self.N):
            for j in range(self.M):
                B[i,j] = np.abs(self.lam[j])**i*np.exp(1.0j*i*np.angle(self.lam[j]))

        r, _, _, _ = la.lstsq(B, self.X)

        # Extract amplitudes and phases from residues
        self.amps = np.abs(r)
        self.faze = np.angle(r)

        return

    def ReconstructSignal(self, t):
        """
        Having computed the amplitudes, damping, frequencies, and phases, can
        produce signal based on sum of series of complex exponentials

        Parameters
        ----------
        t : numpy.ndarray
            array of times

        Returns
        -------
        X : numpy.ndarray
            reconstructed signal

        """
        X = np.zeros(t.shape)
        for i in range(self.M):
            a = self.amps[i]
            x = self.damp[i]
            w = self.freq[i]
            p = self.faze[i]

            X += a*np.exp(x*(t - self.t0))*np.cos(w*(t - self.t0) + p)

        return X
