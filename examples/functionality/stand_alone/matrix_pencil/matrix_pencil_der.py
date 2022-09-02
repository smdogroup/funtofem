import numpy as np
from scipy import linalg as la


def c_ks(alphas, rho):
    """
    Kreisselmeier-Steinhauser (KS) function to approximate maximum real
    part of exponent

    Returns
    -------
    float
        approximate maximum of real part of exponents

    """
    m = alphas.min()

    return -m + np.log(np.sum(np.exp(-rho * (alphas - m)))) / rho


def DcDalpha(alphas, rho):
    """
    Derivative of KS function with respect to the input array of exponents

    Parameters
    ----------
    alphas : numpy.ndarray
        real part of exponents computed from matrix pencil
    rho : float
        KS parameter

    Returns
    -------
    numpy.ndarray
        array of derivatives

    """
    m = alphas.max()
    a = np.sum(np.exp(rho * (alphas - m)))

    return np.exp(rho * (alphas - m)) / a


def DalphaDlam(lam, dt):
    """
    Derivative of exponents with respect to eigenvalues

    Parameters
    ----------
    lam : numpy.ndarray
        eigenvalues
    dt : float
        time step

    Returns
    -------
    numpy.ndarray
        derivatives

    """
    real_part = (1.0 / dt) * lam.real / np.real(np.conj(lam) * lam)
    imag_part = (1.0 / dt) * lam.imag / np.real(np.conj(lam) * lam)

    return real_part + 1j * imag_part


def DlamDA(A):
    """
    Derivatives of each eigenvalue with respect to originating matrix

    Parameters
    ----------
    A : numpy.ndarray
        matrix

    Returns
    -------
    dlam : numpy.ndarray
        matrix of derivatives

    """
    lam, W, V = la.eig(A, left=True, right=True)
    WH = W.conj().T
    m = len(lam)
    dlam = np.zeros((m, m, m), dtype=lam.dtype)
    for i in range(m):
        w = WH[i, :]
        v = V[:, i]
        norm = w.dot(v)
        dlam[i, :, :] = np.outer(w, v) / norm

    return dlam


def SVDDerivative(U, s, VT):
    """
    Derivatives of SVD of full-rank rectangular matrix of size m x n

    Parameters
    ----------
    U : numpy.ndarray
        left singular vectors
    s : numpy.ndarray
        singular values
    VT : numpy.ndarray
        right singular vectors

    Returns
    -------
    dU : numpy.ndarray
        derivatives dU[i,j]/dA[k,l]
    ds : numpy.ndarray
        derivatives ds[i]/dA[k,l]
    dVT : numpy.ndarray
        derivatives dVT[i,j]/dA[k,l]

    Notes
    -----
    This function does not address the case of degenerate SVDs. It expects that
    no two singular values will be identical

    You can find an explanation for the algorithm here at:
    http://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf

    """
    m = U.shape[0]
    n = VT.shape[1]
    ns = len(s)

    # Allocate output arrays
    dU = np.zeros((m, m, m, n))
    ds = np.zeros((ns, m, n))
    dVT = np.zeros((n, n, m, n))

    # Square matrix of singular values
    S1 = np.diag(s)
    S1inv = np.diag(1.0 / s)

    # Form skew-symmetric F matrix
    F = np.zeros((ns, ns))
    for i in range(ns):
        for j in range(i + 1, ns):
            F[i, j] = 1.0 / (s[j] ** 2 - s[i] ** 2)
            F[j, i] = 1.0 / (s[i] ** 2 - s[j] ** 2)

    for k in range(m):
        for l in range(n):
            dP = np.outer(U[k, :], VT[:, l])

            # Extract diagonal for ds
            ds[:, k, l] = np.diag(dP)

            # Compute dC and dD matrices for various cases
            if m > n:
                dP1 = dP[:n, :]
                dP2 = dP[n:, :]

                dC1 = F * (dP1.dot(S1) + S1.dot(dP1.T))
                dDT = -F * (S1.dot(dP1) + dP1.T.dot(S1))

                dC2T = dP2.dot(S1inv)

                dC = np.zeros((m, m))
                dC[:n, :n] = dC1
                dC[:n, n:] = -dC2T.T
                dC[n:, :n] = dC2T

            else:
                dP1 = dP[:, :m]
                dP2 = dP[:, m:]

                dC = F * (dP1.dot(S1) + S1.dot(dP1.T))
                dD1 = F * (S1.dot(dP1) + dP1.T.dot(S1))

                dD2 = S1inv.dot(dP2)

                if m == n:
                    dDT = -dD1
                else:
                    dDT = np.zeros((n, n))
                    dDT[:m, :m] = -dD1
                    dDT[:m, m:] = dD2
                    dDT[m:, :m] = -dD2.T

            # Compute dU and dVT sensitivities from dC and dD
            dU[:, :, k, l] = U.dot(dC)
            dVT[:, :, k, l] = dDT.dot(VT)

    return dU, ds, dVT


def PseudoinverseDerivative(A, Ainv):
    """
    Derivatives of pseudoinverse with respect to its generating matrix

    Parameters
    ----------
    A : numpy.ndarray
        input matrix
    Ainv : numpy.ndarray
        Pseudoinverse of A matrix

    Returns
    -------
    dAinv : numpy.ndarray
        derivatives dAinv[i,j]/dA[k,l]

    """
    m = A.shape[0]
    n = A.shape[1]

    # Allocate array for output
    dAinv = np.zeros((n, m, m, n))

    for k in range(m):
        for l in range(n):
            ek = np.zeros(m)
            ek[k] += 1.0
            el = np.zeros(n)
            el[l] += 1.0

            dA = np.outer(ek, el)

            dAinv[:, :, k, l] = (
                -Ainv.dot(dA).dot(Ainv)
                + Ainv.dot(Ainv.T).dot(dA.T).dot(np.eye(m) - A.dot(Ainv))
                + (np.eye(n) - Ainv.dot(A)).dot(dA.T).dot(Ainv.T).dot(Ainv)
            )

    return dAinv


def DalphaDlamTrans(dcda, lam, dt):
    """
    Apply action of [d(alpha)/d(lam)]^{T} to the vector of derivatives
    [d(c)/d(alpha)]^{T} to obtain the derivatives d(c)/d(lam)

    Parameters
    ----------
    dcda : numpy.ndarray
        vector of derivatives d(c)/d(alpha)
    lam : numpy.ndarray
        eigenvalues of A matrix
    dt : float
        time step

    Returns
    -------
    numpy.ndarray
        vector of derivatives d(c)/d(lam)

    """
    M = dcda.shape[0]
    L = lam.shape[0]

    # Pad the dcda derivative with zeros
    if M < L:
        dcda = np.hstack((dcda, np.zeros(L - M)))

    dadl_real = (1.0 / dt) * lam.real / np.real(np.conj(lam) * lam)
    dadl_imag = (1.0 / dt) * lam.imag / np.real(np.conj(lam) * lam)

    return dcda * dadl_real + 1j * dcda * dadl_imag


def DlamDATrans(dcdl, W, V):
    """
    Apply action of [d(lam)/d(A)]^{T} to the vector of derivatives
    [d(c)/d(lam)]^{T} to obtain the derivatives d(c)/d(A)

    Parameters
    ----------
    dcdl : numpy.ndarray
        vector of derivatives d(c)/d(lam)
    W : numpy.ndarray
        left eigenvectors of matrix A
    V : numpy.ndarray
        right eigenvectors of matrix A

    Returns
    -------
    dcdA : numpy.ndarray
        vector of derivatives d(c)/d(A)

    """
    WH = W.conj().T
    m = len(dcdl)
    dcdA = np.zeros((m, m))
    for i in range(m):
        w = WH[i, :]
        v = V[:, i]
        norm = w.dot(v)
        dldA = np.outer(w, v) / norm
        dcdA += dcdl[i].real * dldA.real + dcdl[i].imag * dldA.imag

    return dcdA


def PseudoInverseDeriv(dcdA, A, Ainv):
    AAinv = la.inv(np.dot(A, A.T))
    T1 = np.dot(np.dot(AAinv, dcdA.T), (np.eye(A.shape[1]) - np.dot(Ainv, A)))
    T2 = np.dot(Ainv.T, np.dot(dcdA, Ainv.T))

    return T1 - T2


def dAdV1Trans(dcdA, V1T, V1inv, V2T):
    """
    Apply action of [d(A)/d(V1^{T})]^{T} to the array of derivatives
    [d(c)/d(A)]^{T} to obtain the derivatives d(c)/d(V1^{T})

    Parameters
    ----------
    dcdA : numpy.ndarray
        array of derivatives d(c)/d(A)
    V1T : numpy.ndarray
        filtered right singular vectors of Y1 matrix
    Vinv : numpy.ndarray
        pseudoinverse of V1T matrix
    V2T : numpy.ndarray
        filtered right singular vectors of Y2 matrix

    Returns
    -------
    dcdV1T : numpy.ndarray
        vector of derivatives d(c)/d(V1^{T})

    """

    # Compute dcdV1inv derivative
    dcdV1inv = (V2T.T).dot(dcdA)

    return PseudoInverseDeriv(dcdV1inv, V1T, V1inv)


def dAdV2Trans(dcdA, V1inv):
    """
    Apply action of [d(A)/d(V2^{T})]^{T} to the array of derivatives
    [d(c)/d(A)]^{T} to obtain the derivatives d(c)/d(V2^{T})

    Parameters
    ----------
    dcdA : numpy.ndarray
        array of derivatives d(c)/d(A)
    V1inv : numpy.ndarray
        generalized inverse of the tranpose of the V1hat matrix

    Returns
    -------
    dcdV2T : numpy.ndarray
        vector of derivatives d(c)/d(V2^{T})

    """

    dcdV2T = np.dot(dcdA, V1inv.T)

    return dcdV2T


def dV12dVhatTrans(dcdV1T, dcdV2T):
    """
    Pad the d(c)/d(V1^{T}) derivatives  with zeros, pad the d(c)/d(V2^{T})
    derivatives with zeros, and combine to obtain the derivatives
    d(c)/d(Vhat^{T})

    Parameters
    ----------
    dcdV1T : numpy.ndarray
        vector of derivatives d(c)/d(V1^{T})
    dcdV2T : numpy.ndarray
        vector of derivatives d(c)/d(V2^{T})

    Returns
    -------
    dcdVhat : numpy.ndarray
        vector of derivatives d(c)/d(Vhat^{T})

    """
    M = dcdV1T.shape[0]
    dcdVhat = np.hstack((dcdV1T, np.zeros(M).reshape((M, 1)))) + np.hstack(
        (np.zeros(M).reshape((M, 1)), dcdV2T)
    )

    return dcdVhat


def dVhatdYTrans(dcdVhat, U, s, VT):
    """
    Apply action of [d(Vhat^{T})/d(Y)]^{T} to the array of derivatives
    [d(c)/d(Vhat^{T})]^{T} to obtain the derivatives d(c)/d(Y)

    Parameters
    ----------
    dcdVhat : numpy.ndarray
        array of derivatives d(c)/d(Vhat)
    U : numpy.ndarray
        left singular vectors
    s : numpy.ndarray
        singular values
    VT : numpy.ndarray
        right singular vectors

    Returns
    -------
    dU : numpy.ndarray
        derivatives dU[i,j]/dA[k,l]

    Returns
    -------
    dcdY : numpy.ndarray
        array of derivatives d(c)/d(Y)

    """
    m = U.shape[0]
    n = VT.shape[1]
    M = dcdVhat.shape[0]

    # Compute the entries of the skew symmetric matrix
    F = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            F[i, j] = 1.0 / (s[j] ** 2 - s[i] ** 2)
            F[j, i] = 1.0 / (s[i] ** 2 - s[j] ** 2)

    # Compute the E1 matrix
    E1 = np.zeros((n, m))
    for i in range(n):
        E1[i, i] = 1.0

    dcdV = np.zeros((n, n))
    dcdV[:, :M] = dcdVhat.T

    # Compute the derivative
    B = F * (np.dot(VT, dcdV) - np.dot(dcdV.T, VT.T))
    dcdY = np.dot(U, np.dot(E1.T, np.dot(np.diag(s), np.dot(B, VT))))

    return dcdY


def dYdXTrans(dcdY):
    """
    Apply action of [d(Y)/d(X)]^{T} to the array of derivatives [d(c)/d(Y)]^{T}
    to obtain the derivatives d(c)/d(X)

    Parameters
    ----------
    dcdY : numpy.ndarray
        array of derivatives d(c)/d(Y)

    Returns
    -------
    dcdX : numpy.ndarray
        array of derivatives d(c)/d(X)

    """
    L = dcdY.shape[1] - 1
    N = dcdY.shape[0] + L

    dcdX = np.zeros(N)

    # Sum the anti-diagonals into dcdX
    for i in range(N - L):
        for j in range(L + 1):
            dcdX[i + j] += dcdY[i, j]

    return dcdX
