/*
  This file is part of the package FUNtoFEM for coupled aeroelastic simulation
  and design optimization.

  Copyright (C) 2015 Georgia Tech Research Corporation.
  Additional copyright (C) 2015 Kevin Jacobson, Jan Kiviaho and Graeme Kennedy.
  All rights reserved.

  FUNtoFEM is licensed under the Apache License, Version 2.0 (the "License");
  you may not use this software except in compliance with the License.
  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/

#include "LinearizedMELD.h"

#include <stdio.h>

#include <cstring>

#include "MELD.h"
#include "funtofemlapack.h"

LinearizedMELD::LinearizedMELD(MPI_Comm all, MPI_Comm structure,
                               int struct_root, MPI_Comm aero, int aero_root,
                               int symmetry, int num_nearest, F2FScalar beta)
    : MELD(all, structure, struct_root, aero, aero_root, symmetry, num_nearest,
           beta) {
  // Initialize the data for the transfers
  global_H = NULL;

  // Notify user of the type of transfer scheme they are using
  int rank;
  MPI_Comm_rank(global_comm, &rank);
  if (rank == struct_root) {
    printf("Transfer scheme [%i]: Creating scheme of type LinearizedMELD...\n",
           object_id);
  }
}

LinearizedMELD::~LinearizedMELD() {
  // Free the data for the transfers
  if (global_H) {
    delete[] global_H;
  }

  int rank;
  MPI_Comm_rank(global_comm, &rank);
  if (rank == struct_root) {
    printf("Transfer scheme [%i]: freeing LinearizedMELD data...\n", object_id);
  }
}

/*
  Set aerostructural connectivity, compute weights, and allocate memory needed
  for transfers and products

  Arguments
  ---------
  num_nearest : the number of struct nodes connected to each aero node
  global_beta : the weighting decay parameter
*/
void LinearizedMELD::initialize() {
  // global number of structural nodes
  distributeStructuralMesh();

  // Check that user doesn't set more nearest nodes than exist in total
  if (nn > ns) {
    nn = ns;
  }

  if (Us) {
    delete[] Us;
  }
  Us = NULL;
  if (Fa) {
    delete[] Fa;
  }
  Fa = NULL;

  if (na > 0) {
    Fa = new F2FScalar[3 * na];
    memset(Fa, 0, 3 * na * sizeof(F2FScalar));
  }
  if (ns > 0) {
    Us = new F2FScalar[3 * ns];
    memset(Us, 0, 3 * ns * sizeof(F2FScalar));
  }

  // Create aerostructural connectivity
  global_conn = new int[nn * na];
  computeAeroStructConn(isymm, nn, global_conn);

  // Allocate and compute the weights
  global_W = new F2FScalar[nn * na];
  computeWeights(F2FRealPart(global_beta), isymm, nn, global_conn, global_W);

  // Allocate transfer variables
  global_xs0bar = new F2FScalar[3 * na];
  global_H = new F2FScalar[9 * na];
}

/*
  Computes the displacements of all aerodynamic surface nodes based on
  linearized version of MELD

  Arguments
  ---------
  struct_disps : structural node displacements

  Returns
  -------
  aero_disps   : aerodynamic node displacements
*/
void LinearizedMELD::transferDisps(const F2FScalar *struct_disps,
                                   F2FScalar *aero_disps) {
  // Check if struct nodes locations need to be redistributed
  distributeStructuralMesh();

  // Copy prescribed displacements into displacement vector
  structGatherBcast(3 * ns_local, struct_disps, 3 * ns, Us);

  // Zero the outputs
  memset(aero_disps, 0, 3 * na * sizeof(F2FScalar));

  for (int i = 0; i < na; i++) {
    // Point aerodynamic surface node location into a
    const F2FScalar *xa = &Xa[3 * i];

    // Compute the centroid of the initial set of nodes
    const int *local_conn = &global_conn[i * nn];
    const F2FScalar *W = &global_W[i * nn];
    F2FScalar *xs0bar = &global_xs0bar[3 * i];
    computeCentroid(local_conn, W, Xs, xs0bar);

    // Compute the covariance matrix
    F2FScalar *H = &global_H[9 * i];
    computeCovariance(Xs, Xs, local_conn, W, xs0bar, xs0bar, H);

    // Compute the inverse of the point inertia matrix
    F2FScalar Hinv[9];
    computePointInertiaInverse(H, Hinv);

    // Form the vector r from the initial centroid to the aerodynamic surface
    // node
    F2FScalar r[3];
    vec_diff(xs0bar, xa, r);

    // Sum the contributions from each structural node to the aerodynamic
    // surface node's displacement
    F2FScalar *ua = &aero_disps[3 * i];

    for (int j = 0; j < nn; j++) {
      // Get structural node location
      int indx = global_conn[nn * i + j];

      if (indx < ns) {
        F2FScalar *xs = &Xs[3 * indx];

        // Form the vector q from the centroid of the undisplaced set to the
        // node
        F2FScalar q[3];
        vec_diff(xs0bar, xs, q);

        // Get structural node displacement
        const F2FScalar *us = &Us[3 * indx];

        // Compute and add contribution to aerodynamic surface node displacement
        F2FScalar w = W[j];
        F2FScalar uj[3];
        computeDispContribution(w, r, Hinv, q, us, uj);
        vec_add(uj, ua, ua);
      } else {
        indx = indx - ns;

        // Form the vector q from the centroid of the undisplaced set to the
        // node
        const F2FScalar *xs0 = &Xs[3 * indx];
        F2FScalar rxs0[3];
        memcpy(rxs0, xs0, 3 * sizeof(F2FScalar));
        rxs0[isymm] *= -1.0;

        F2FScalar q[3];
        vec_diff(xs0bar, rxs0, q);

        // Get structural node displacement
        const F2FScalar *us = &Us[3 * indx];

        // Compute and add contribution to aerodynamic surface node displacement
        F2FScalar w = W[j];
        F2FScalar uj[3];
        computeDispContribution(w, r, Hinv, q, us, uj);
        vec_add(uj, ua, ua);
      }
    }
  }
}

/*
  Computes inverse of point inertia matrix Hbar = H - I*Tr(H)

  Arguments
  ---------
  H    : covariance matrix

  Returns
  -------
  Hinv : inverse of point inertia matrix
*/
void LinearizedMELD::computePointInertiaInverse(const F2FScalar *H,
                                                F2FScalar *Hinv) {
  // Make a copy of H
  F2FScalar Hcopy[9];
  memcpy(Hcopy, H, 9 * sizeof(F2FScalar));

  // Compute Hbar = H - I*Tr(H) and store in Hcopy
  F2FScalar trace = Hcopy[0] + Hcopy[4] + Hcopy[8];
  Hcopy[0] -= trace;
  Hcopy[4] -= trace;
  Hcopy[8] -= trace;

  // Set Hinv = I
  memset(Hinv, 0, 9 * sizeof(F2FScalar));
  Hinv[0] = 1.0;
  Hinv[4] = 1.0;
  Hinv[8] = 1.0;

#ifdef FUNTOFEM_USE_COMPLEX
  int n = 3;             // Dimension of all the matrices
  double s[3];           // Singular values
  double rcond = 1e-10;  // Used to determine the effective rank
  int rank = 0;          // Rank of the matrix that is used as output
  F2FScalar work[15];    // Work array
  int lwork = 5 * 3;     // = 5 * n
  double rwork[15];
  int info = 0;
  LAPACKzgelss(&n, &n, &n, Hcopy, &n, Hinv, &n, s, &rcond, &rank, work, &lwork,
               rwork, &info);
#else
  int n = 3;             // Dimension of all the matrices
  double s[3];           // Singular values
  double rcond = 1e-10;  // Used to determine the effective rank
  int rank = 0;          // Rank of the matrix that is used as output
  double work[15];       // Work array
  int lwork = 5 * 3;     // = 5 * n
  int info = 0;
  LAPACKdgelss(&n, &n, &n, Hcopy, &n, Hinv, &n, s, &rcond, &rank, work, &lwork,
               &info);
#endif  // FUNTOFEM_USE_COMPLEX

  // Find the least squares inverse of the matrix
  if (info) {
    printf("LinearizedMELD error: Least squares solution for H failed\n");
  }
}

/*
  Compute the result of the

*/
void LinearizedMELD::adjPointInertiaInverse(const F2FScalar *Hinv,
                                            const F2FScalar *Hinvd,
                                            F2FScalar *Hd) {
  F2FScalar A[9];
  // A = Hinv^{T} * Hinvd
  A[0] = Hinv[0] * Hinvd[0] + Hinv[3] * Hinvd[3] + Hinv[6] * Hinvd[6];
  A[3] = Hinv[1] * Hinvd[0] + Hinv[4] * Hinvd[3] + Hinv[7] * Hinvd[6];
  A[6] = Hinv[2] * Hinvd[0] + Hinv[5] * Hinvd[3] + Hinv[8] * Hinvd[6];

  A[1] = Hinv[0] * Hinvd[1] + Hinv[3] * Hinvd[4] + Hinv[6] * Hinvd[7];
  A[4] = Hinv[1] * Hinvd[1] + Hinv[4] * Hinvd[4] + Hinv[7] * Hinvd[7];
  A[7] = Hinv[2] * Hinvd[1] + Hinv[5] * Hinvd[4] + Hinv[8] * Hinvd[7];

  A[2] = Hinv[0] * Hinvd[2] + Hinv[3] * Hinvd[5] + Hinv[6] * Hinvd[8];
  A[5] = Hinv[1] * Hinvd[2] + Hinv[4] * Hinvd[5] + Hinv[7] * Hinvd[8];
  A[8] = Hinv[2] * Hinvd[2] + Hinv[5] * Hinvd[5] + Hinv[8] * Hinvd[8];

  // Hd = - A * Hinv^{T}
  Hd[0] = -(A[0] * Hinv[0] + A[1] * Hinv[1] + A[2] * Hinv[2]);
  Hd[3] = -(A[3] * Hinv[0] + A[4] * Hinv[1] + A[5] * Hinv[2]);
  Hd[6] = -(A[6] * Hinv[0] + A[7] * Hinv[1] + A[8] * Hinv[2]);

  Hd[1] = -(A[0] * Hinv[3] + A[1] * Hinv[4] + A[2] * Hinv[5]);
  Hd[4] = -(A[3] * Hinv[3] + A[4] * Hinv[4] + A[5] * Hinv[5]);
  Hd[7] = -(A[6] * Hinv[3] + A[7] * Hinv[4] + A[8] * Hinv[5]);

  Hd[2] = -(A[0] * Hinv[6] + A[1] * Hinv[7] + A[2] * Hinv[8]);
  Hd[5] = -(A[3] * Hinv[6] + A[4] * Hinv[7] + A[5] * Hinv[8]);
  Hd[8] = -(A[6] * Hinv[6] + A[7] * Hinv[7] + A[8] * Hinv[8]);

  // Hd = Hd - trace(Hd)
  F2FScalar trace = Hd[0] + Hd[4] + Hd[8];
  Hd[0] = Hd[0] - trace;
  Hd[4] = Hd[4] - trace;
  Hd[8] = Hd[8] - trace;
}

/*
  Computes contribution to displacement of aerodynamic surface node from single
  structural node in linearized MELD

  Arguments
  ---------
  w    : weight of structural node
  r    : vector from centroid to aerodynamic surface node
  Hinv : inverse of point inertia matrix
  q    : vector from centroid to structural node
  us   : displacement of structural node

  Returns
  -------
  ua   : contribution to displacement of aerodynamic surface node
*/
void LinearizedMELD::computeDispContribution(
    const F2FScalar w, const F2FScalar *r, const F2FScalar *Hinv,
    const F2FScalar *q, const F2FScalar *us, F2FScalar *ua) {
  // Compute matrix = w*(q^{x} * Hinv * r^{x} + I)
  F2FScalar A[9];
  A[0] = w * (q[2] * (r[1] * Hinv[5] - r[2] * Hinv[4]) -
              q[1] * (r[1] * Hinv[8] - r[2] * Hinv[7]) + 1.0);
  A[1] = w * (q[1] * (r[0] * Hinv[8] - r[2] * Hinv[6]) -
              q[2] * (r[0] * Hinv[5] - r[2] * Hinv[3]));
  A[2] = w * (q[2] * (r[0] * Hinv[4] - r[1] * Hinv[3]) -
              q[1] * (r[0] * Hinv[7] - r[1] * Hinv[6]));
  A[3] = w * (q[0] * (r[1] * Hinv[8] - r[2] * Hinv[7]) -
              q[2] * (r[1] * Hinv[2] - r[2] * Hinv[1]));
  A[4] = w * (q[2] * (r[0] * Hinv[2] - r[2] * Hinv[0]) -
              q[0] * (r[0] * Hinv[8] - r[2] * Hinv[6]) + 1.0);
  A[5] = w * (q[0] * (r[0] * Hinv[7] - r[1] * Hinv[6]) -
              q[2] * (r[0] * Hinv[1] - r[1] * Hinv[0]));
  A[6] = w * (q[1] * (r[1] * Hinv[2] - r[2] * Hinv[1]) -
              q[0] * (r[1] * Hinv[5] - r[2] * Hinv[4]));
  A[7] = w * (q[0] * (r[0] * Hinv[5] - r[2] * Hinv[3]) -
              q[1] * (r[0] * Hinv[2] - r[2] * Hinv[0]));
  A[8] = w * (q[1] * (r[0] * Hinv[1] - r[1] * Hinv[0]) -
              q[0] * (r[0] * Hinv[4] - r[1] * Hinv[3]) + 1.0);

  // Compute matrix-vector product ua = A*us
  ua[0] = A[0] * us[0] + A[3] * us[1] + A[6] * us[2];
  ua[1] = A[1] * us[0] + A[4] * us[1] + A[7] * us[2];
  ua[2] = A[2] * us[0] + A[5] * us[1] + A[8] * us[2];
}

/*
  Compute the contributions to the adjoint from the displacement
*/
void LinearizedMELD::addAdjDispContribution(
    const F2FScalar w, const F2FScalar *r, const F2FScalar *Hinv,
    const F2FScalar *q, const F2FScalar *us, const F2FScalar *uad,
    F2FScalar *rd, F2FScalar *qd, F2FScalar *Hinvd) {
  // expr = w * uad^{T} * q^{x} * Hinv * r^{x} * us = - w * uad^{T} * qx * Hinv
  // * us^{x} * r rd = d(expr)/dr = - w * uad^{T} * qx * Hinv * us^{x}
  F2FScalar Ad[9];
  Ad[0] = uad[0] * us[0];
  Ad[1] = uad[1] * us[0];
  Ad[2] = uad[2] * us[0];

  Ad[3] = uad[0] * us[1];
  Ad[4] = uad[1] * us[1];
  Ad[5] = uad[2] * us[1];

  Ad[6] = uad[0] * us[2];
  Ad[7] = uad[1] * us[2];
  Ad[8] = uad[2] * us[2];

  if (rd) {
    rd[0] += w * (-Ad[1] * (Hinv[5] * q[2] - Hinv[8] * q[1]) +
                  Ad[2] * (Hinv[4] * q[2] - Hinv[7] * q[1]) +
                  Ad[4] * (Hinv[2] * q[2] - Hinv[8] * q[0]) -
                  Ad[5] * (Hinv[1] * q[2] - Hinv[7] * q[0]) -
                  Ad[7] * (Hinv[2] * q[1] - Hinv[5] * q[0]) +
                  Ad[8] * (Hinv[1] * q[1] - Hinv[4] * q[0]));
    rd[1] += w * (Ad[0] * (Hinv[5] * q[2] - Hinv[8] * q[1]) -
                  Ad[2] * (Hinv[3] * q[2] - Hinv[6] * q[1]) -
                  Ad[3] * (Hinv[2] * q[2] - Hinv[8] * q[0]) +
                  Ad[5] * (Hinv[0] * q[2] - Hinv[6] * q[0]) +
                  Ad[6] * (Hinv[2] * q[1] - Hinv[5] * q[0]) -
                  Ad[8] * (Hinv[0] * q[1] - Hinv[3] * q[0]));
    rd[2] += w * (-Ad[0] * (Hinv[4] * q[2] - Hinv[7] * q[1]) +
                  Ad[1] * (Hinv[3] * q[2] - Hinv[6] * q[1]) +
                  Ad[3] * (Hinv[1] * q[2] - Hinv[7] * q[0]) -
                  Ad[4] * (Hinv[0] * q[2] - Hinv[6] * q[0]) -
                  Ad[6] * (Hinv[1] * q[1] - Hinv[4] * q[0]) +
                  Ad[7] * (Hinv[0] * q[1] - Hinv[3] * q[0]));
  }

  if (qd) {
    qd[0] += w * (-Ad[3] * (Hinv[7] * r[2] - Hinv[8] * r[1]) +
                  Ad[4] * (Hinv[6] * r[2] - Hinv[8] * r[0]) -
                  Ad[5] * (Hinv[6] * r[1] - Hinv[7] * r[0]) +
                  Ad[6] * (Hinv[4] * r[2] - Hinv[5] * r[1]) -
                  Ad[7] * (Hinv[3] * r[2] - Hinv[5] * r[0]) +
                  Ad[8] * (Hinv[3] * r[1] - Hinv[4] * r[0]));
    qd[1] += w * (Ad[0] * (Hinv[7] * r[2] - Hinv[8] * r[1]) -
                  Ad[1] * (Hinv[6] * r[2] - Hinv[8] * r[0]) +
                  Ad[2] * (Hinv[6] * r[1] - Hinv[7] * r[0]) -
                  Ad[6] * (Hinv[1] * r[2] - Hinv[2] * r[1]) +
                  Ad[7] * (Hinv[0] * r[2] - Hinv[2] * r[0]) -
                  Ad[8] * (Hinv[0] * r[1] - Hinv[1] * r[0]));
    qd[2] += w * (-Ad[0] * (Hinv[4] * r[2] - Hinv[5] * r[1]) +
                  Ad[1] * (Hinv[3] * r[2] - Hinv[5] * r[0]) -
                  Ad[2] * (Hinv[3] * r[1] - Hinv[4] * r[0]) +
                  Ad[3] * (Hinv[1] * r[2] - Hinv[2] * r[1]) -
                  Ad[4] * (Hinv[0] * r[2] - Hinv[2] * r[0]) +
                  Ad[5] * (Hinv[0] * r[1] - Hinv[1] * r[0]));
  }

  if (Hinvd) {
    Hinvd[0] += w * (-Ad[4] * q[2] * r[2] + Ad[5] * q[2] * r[1] +
                     Ad[7] * q[1] * r[2] - Ad[8] * q[1] * r[1]);
    Hinvd[1] += w * (Ad[3] * q[2] * r[2] - Ad[5] * q[2] * r[0] -
                     Ad[6] * q[1] * r[2] + Ad[8] * q[1] * r[0]);
    Hinvd[2] += w * (-Ad[3] * q[2] * r[1] + Ad[4] * q[2] * r[0] +
                     Ad[6] * q[1] * r[1] - Ad[7] * q[1] * r[0]);
    Hinvd[3] += w * (Ad[1] * q[2] * r[2] - Ad[2] * q[2] * r[1] -
                     Ad[7] * q[0] * r[2] + Ad[8] * q[0] * r[1]);
    Hinvd[4] += w * (-Ad[0] * q[2] * r[2] + Ad[2] * q[2] * r[0] +
                     Ad[6] * q[0] * r[2] - Ad[8] * q[0] * r[0]);
    Hinvd[5] += w * (Ad[0] * q[2] * r[1] - Ad[1] * q[2] * r[0] -
                     Ad[6] * q[0] * r[1] + Ad[7] * q[0] * r[0]);
    Hinvd[6] += w * (-Ad[1] * q[1] * r[2] + Ad[2] * q[1] * r[1] +
                     Ad[4] * q[0] * r[2] - Ad[5] * q[0] * r[1]);
    Hinvd[7] += w * (Ad[0] * q[1] * r[2] - Ad[2] * q[1] * r[0] -
                     Ad[3] * q[0] * r[2] + Ad[5] * q[0] * r[0]);
    Hinvd[8] += w * (-Ad[0] * q[1] * r[1] + Ad[1] * q[1] * r[0] +
                     Ad[3] * q[0] * r[1] - Ad[4] * q[0] * r[0]);
  }
}

/*
  Computes the loads on all structural nodes based on linearized version of
  MELD

  Arguments
  ---------
  aero_loads   : loads on aerodynamic surface nodes

  Returns
  -------
  struct_loads : loads on structural nodes
*/
void LinearizedMELD::transferLoads(const F2FScalar *aero_loads,
                                   F2FScalar *struct_loads) {
  // Copy prescribed aero loads into member variable
  memcpy(Fa, aero_loads, 3 * na * sizeof(F2FScalar));

  // Zero struct loads
  F2FScalar *struct_loads_global = new F2FScalar[3 * ns];
  memset(struct_loads_global, 0, 3 * ns * sizeof(F2FScalar));

  // Loop over aerodynamic surface nodes
  for (int i = 0; i < na; i++) {
    // Get aerodynamic surface node location and load
    F2FScalar *a = &Xa[3 * i];
    const F2FScalar *fa = &Fa[3 * i];

    // Compute vector d from centroid to aerodynamic surface node
    F2FScalar *xs0bar = &global_xs0bar[3 * i];
    F2FScalar r[3];
    vec_diff(xs0bar, a, r);

    // Compute inverse of point inertia matrix
    F2FScalar Hinv[9];
    computePointInertiaInverse(&global_H[9 * i], Hinv);

    // Loop over linked structural nodes
    for (int j = 0; j < nn; j++) {
      // Get structural node using index from connectivity
      int indx = global_conn[nn * i + j];

      if (indx < ns) {
        F2FScalar *xs = &Xs[3 * indx];

        // Compute vector q from centroid to structural node
        F2FScalar q[3];
        vec_diff(xs0bar, xs, q);

        // Compute load contribution
        F2FScalar *fs = &struct_loads_global[3 * indx];
        F2FScalar w = global_W[nn * i + j];
        F2FScalar fj[3];
        computeLoadContribution(w, q, Hinv, r, fa, fj);

        // Add load contribution into global structural load array
        fs[0] += fj[0];
        fs[1] += fj[1];
        fs[2] += fj[2];
      } else {
        indx = indx - ns;

        // Compute vector q from centroid to structural node using the
        // symmetry condition
        const F2FScalar *xs0 = &Xs[3 * indx];
        F2FScalar rxs0[3];
        memcpy(rxs0, xs0, 3 * sizeof(F2FScalar));
        rxs0[isymm] *= -1.0;

        // Compute vector q from centroid to structural node
        F2FScalar q[3];
        vec_diff(xs0bar, rxs0, q);

        // Compute load contribution
        F2FScalar *fs = &struct_loads_global[3 * indx];
        F2FScalar w = global_W[nn * i + j];
        F2FScalar fj[3];
        computeLoadContribution(w, q, Hinv, r, fa, fj);

        // Add load contribution into global structural load array
        fs[0] += fj[0];
        fs[1] += fj[1];
        fs[2] += fj[2];
      }
    }
  }

  // distribute the structural loads
  structAddScatter(3 * ns, struct_loads_global, 3 * ns_local, struct_loads);

  delete[] struct_loads_global;
}

/*
  Computes contribution to displacement of aerodynamic surface node from single
  structural node in linearized MELD

  Arguments
  ---------
  w    : weight of structural node
  q    : vector from centroid to structural node
  Hinv : inverse of point inertia matrix
  r    : vector from centroid to aerodynamic surface node
  fa   : load on aerodynamic surface node

  Returns
  -------
  fj   : contribution to load on structural node
*/
void LinearizedMELD::computeLoadContribution(
    const F2FScalar w, const F2FScalar *q, const F2FScalar *Hinv,
    const F2FScalar *r, const F2FScalar *fa, F2FScalar *fj) {
  // Compute matrix = w*(rx*Hinv*qx + I)
  F2FScalar A[9];

  A[0] = w * (r[1] * (Hinv[7] * q[2] - Hinv[8] * q[1]) -
              r[2] * (Hinv[4] * q[2] - Hinv[5] * q[1]) + 1.0);
  A[1] = w * (r[2] * (Hinv[3] * q[2] - Hinv[5] * q[0]) -
              r[1] * (Hinv[6] * q[2] - Hinv[8] * q[0]));
  A[2] = w * (r[1] * (Hinv[6] * q[1] - Hinv[7] * q[0]) -
              r[2] * (Hinv[3] * q[1] - Hinv[4] * q[0]));
  A[3] = w * (r[2] * (Hinv[1] * q[2] - Hinv[2] * q[1]) -
              r[0] * (Hinv[7] * q[2] - Hinv[8] * q[1]));
  A[4] = w * (r[0] * (Hinv[6] * q[2] - Hinv[8] * q[0]) -
              r[2] * (Hinv[0] * q[2] - Hinv[2] * q[0]) + 1.0);
  A[5] = w * (r[2] * (Hinv[0] * q[1] - Hinv[1] * q[0]) -
              r[0] * (Hinv[6] * q[1] - Hinv[7] * q[0]));
  A[6] = w * (r[0] * (Hinv[4] * q[2] - Hinv[5] * q[1]) -
              r[1] * (Hinv[1] * q[2] - Hinv[2] * q[1]));
  A[7] = w * (r[1] * (Hinv[0] * q[2] - Hinv[2] * q[0]) -
              r[0] * (Hinv[3] * q[2] - Hinv[5] * q[0]));
  A[8] = w * (r[0] * (Hinv[3] * q[1] - Hinv[4] * q[0]) -
              r[1] * (Hinv[0] * q[1] - Hinv[1] * q[0]) + 1.0);

  // Compute matrix-vector product ua = A*us
  fj[0] = A[0] * fa[0] + A[3] * fa[1] + A[6] * fa[2];
  fj[1] = A[1] * fa[0] + A[4] * fa[1] + A[7] * fa[2];
  fj[2] = A[2] * fa[0] + A[5] * fa[1] + A[8] * fa[2];
}

/*
  Compute the contributions to the adjoint from the load
*/
void LinearizedMELD::addAdjLoadContribution(
    const F2FScalar w, const F2FScalar *r, const F2FScalar *Hinv,
    const F2FScalar *q, const F2FScalar *fa, const F2FScalar *fjd,
    F2FScalar *rd, F2FScalar *qd, F2FScalar *Hinvd) {
  F2FScalar Ad[9];
  Ad[0] = fjd[0] * fa[0];
  Ad[1] = fjd[1] * fa[0];
  Ad[2] = fjd[2] * fa[0];

  Ad[3] = fjd[0] * fa[1];
  Ad[4] = fjd[1] * fa[1];
  Ad[5] = fjd[2] * fa[1];

  Ad[6] = fjd[0] * fa[2];
  Ad[7] = fjd[1] * fa[2];
  Ad[8] = fjd[2] * fa[2];

  if (rd) {
    rd[0] += w * (-Ad[3] * (Hinv[7] * q[2] - Hinv[8] * q[1]) +
                  Ad[4] * (Hinv[6] * q[2] - Hinv[8] * q[0]) -
                  Ad[5] * (Hinv[6] * q[1] - Hinv[7] * q[0]) +
                  Ad[6] * (Hinv[4] * q[2] - Hinv[5] * q[1]) -
                  Ad[7] * (Hinv[3] * q[2] - Hinv[5] * q[0]) +
                  Ad[8] * (Hinv[3] * q[1] - Hinv[4] * q[0]));
    rd[1] += w * (Ad[0] * (Hinv[7] * q[2] - Hinv[8] * q[1]) -
                  Ad[1] * (Hinv[6] * q[2] - Hinv[8] * q[0]) +
                  Ad[2] * (Hinv[6] * q[1] - Hinv[7] * q[0]) -
                  Ad[6] * (Hinv[1] * q[2] - Hinv[2] * q[1]) +
                  Ad[7] * (Hinv[0] * q[2] - Hinv[2] * q[0]) -
                  Ad[8] * (Hinv[0] * q[1] - Hinv[1] * q[0]));
    rd[2] += w * (-Ad[0] * (Hinv[4] * q[2] - Hinv[5] * q[1]) +
                  Ad[1] * (Hinv[3] * q[2] - Hinv[5] * q[0]) -
                  Ad[2] * (Hinv[3] * q[1] - Hinv[4] * q[0]) +
                  Ad[3] * (Hinv[1] * q[2] - Hinv[2] * q[1]) -
                  Ad[4] * (Hinv[0] * q[2] - Hinv[2] * q[0]) +
                  Ad[5] * (Hinv[0] * q[1] - Hinv[1] * q[0]));
  }
  if (qd) {
    qd[0] += w * (-Ad[1] * (Hinv[5] * r[2] - Hinv[8] * r[1]) +
                  Ad[2] * (Hinv[4] * r[2] - Hinv[7] * r[1]) +
                  Ad[4] * (Hinv[2] * r[2] - Hinv[8] * r[0]) -
                  Ad[5] * (Hinv[1] * r[2] - Hinv[7] * r[0]) -
                  Ad[7] * (Hinv[2] * r[1] - Hinv[5] * r[0]) +
                  Ad[8] * (Hinv[1] * r[1] - Hinv[4] * r[0]));
    qd[1] += w * (Ad[0] * (Hinv[5] * r[2] - Hinv[8] * r[1]) -
                  Ad[2] * (Hinv[3] * r[2] - Hinv[6] * r[1]) -
                  Ad[3] * (Hinv[2] * r[2] - Hinv[8] * r[0]) +
                  Ad[5] * (Hinv[0] * r[2] - Hinv[6] * r[0]) +
                  Ad[6] * (Hinv[2] * r[1] - Hinv[5] * r[0]) -
                  Ad[8] * (Hinv[0] * r[1] - Hinv[3] * r[0]));
    qd[2] += w * (-Ad[0] * (Hinv[4] * r[2] - Hinv[7] * r[1]) +
                  Ad[1] * (Hinv[3] * r[2] - Hinv[6] * r[1]) +
                  Ad[3] * (Hinv[1] * r[2] - Hinv[7] * r[0]) -
                  Ad[4] * (Hinv[0] * r[2] - Hinv[6] * r[0]) -
                  Ad[6] * (Hinv[1] * r[1] - Hinv[4] * r[0]) +
                  Ad[7] * (Hinv[0] * r[1] - Hinv[3] * r[0]));
  }

  if (Hinvd) {
    Hinvd[0] += w * (-Ad[4] * q[2] * r[2] + Ad[5] * q[1] * r[2] +
                     Ad[7] * q[2] * r[1] - Ad[8] * q[1] * r[1]);
    Hinvd[1] += w * (Ad[3] * q[2] * r[2] - Ad[5] * q[0] * r[2] -
                     Ad[6] * q[2] * r[1] + Ad[8] * q[0] * r[1]);
    Hinvd[2] += w * (-Ad[3] * q[1] * r[2] + Ad[4] * q[0] * r[2] +
                     Ad[6] * q[1] * r[1] - Ad[7] * q[0] * r[1]);
    Hinvd[3] += w * (Ad[1] * q[2] * r[2] - Ad[2] * q[1] * r[2] -
                     Ad[7] * q[2] * r[0] + Ad[8] * q[1] * r[0]);
    Hinvd[4] += w * (-Ad[0] * q[2] * r[2] + Ad[2] * q[0] * r[2] +
                     Ad[6] * q[2] * r[0] - Ad[8] * q[0] * r[0]);
    Hinvd[5] += w * (Ad[0] * q[1] * r[2] - Ad[1] * q[0] * r[2] -
                     Ad[6] * q[1] * r[0] + Ad[7] * q[0] * r[0]);
    Hinvd[6] += w * (-Ad[1] * q[2] * r[1] + Ad[2] * q[1] * r[1] +
                     Ad[4] * q[2] * r[0] - Ad[5] * q[1] * r[0]);
    Hinvd[7] += w * (Ad[0] * q[2] * r[1] - Ad[2] * q[0] * r[1] -
                     Ad[3] * q[2] * r[0] + Ad[5] * q[0] * r[0]);
    Hinvd[8] += w * (-Ad[0] * q[1] * r[1] + Ad[1] * q[0] * r[1] +
                     Ad[3] * q[1] * r[0] - Ad[4] * q[0] * r[0]);
  }
}

/*
  Apply the action of the displacement transfer w.r.t structural displacments
  Jacobian to the input vector

  Arguments
  ----------
  vecs  : input vector

  Returns
  --------
  prods : output vector
*/
void LinearizedMELD::applydDduS(const F2FScalar *vecs, F2FScalar *prods) {
  transferDisps(vecs, prods);

  // Reverse sign due to definition of diplacement transfer residual
  for (int i = 0; i < 3 * na; i++) {
    prods[i] *= -1.0;
  }
}

/*
  Apply the action of the displacement transfer w.r.t structural displacements
  transpose Jacobian to the input vector

  Arguments
  ----------
  vecs  : input vector

  Returns
  --------
  prods : output vector
*/
void LinearizedMELD::applydDduSTrans(const F2FScalar *vecs, F2FScalar *prods) {
  transferLoads(vecs, prods);

  // Reverse sign due to definition of diplacement transfer residual
  for (int j = 0; j < 3 * ns_local; j++) {
    prods[j] *= -1.0;
  }
}

/*
  Apply the action of the load transfer w.r.t structural displacements Jacobian
  to the input vector

  Arguments
  ----------
  vecs  : input vector

  Returns
  --------
  prods : output vector
*/
void LinearizedMELD::applydLduS(const F2FScalar *vecs, F2FScalar *prods) {
  memset(prods, 0, 3 * ns_local * sizeof(F2FScalar));
}

/*
  Apply the action of the load transfer w.r.t structural displacements
  transpose Jacobian to the input vector

  Arguments
  ----------
  vecs  : input vector

  Returns
  --------
  prods : output vector
*/
void LinearizedMELD::applydLduSTrans(const F2FScalar *vecs, F2FScalar *prods) {
  memset(prods, 0, 3 * ns_local * sizeof(F2FScalar));
}

/*
  Apply the action of the displacement transfer w.r.t initial aerodynamic
  surface node locations Jacobian to the right of the transposed input vector

  Arguments
  ----------
  vecs  : input vector

  Returns
  --------
  prods : output vector
*/
void LinearizedMELD::applydDdxA0(const F2FScalar *vecs, F2FScalar *prods) {
  for (int i = 0; i < na; i++) {
    // Get vector of adjoint variables and rotation matrix for each aerodynamic
    // node
    const F2FScalar *lam = &vecs[3 * i];

    // Point aerodynamic surface node location into a
    const F2FScalar *xa = &Xa[3 * i];

    // Compute the centroid of the initial set of nodes
    const F2FScalar *W = &global_W[i * nn];
    F2FScalar *xs0bar = &global_xs0bar[3 * i];

    // Compute the inverse of the point inertia matrix
    F2FScalar *H = &global_H[9 * i];
    F2FScalar Hinv[9];
    computePointInertiaInverse(H, Hinv);

    // Form the vector r from the initial centroid to the aerodynamic surface
    // node
    F2FScalar r[3];
    vec_diff(xs0bar, xa, r);

    // Zero the derivative contributions
    F2FScalar rd[3];
    memset(rd, 0, 3 * sizeof(F2FScalar));

    for (int j = 0; j < nn; j++) {
      // Get structural node location
      int indx = global_conn[nn * i + j];

      if (indx < ns) {
        F2FScalar *xs = &Xs[3 * indx];

        // Form the vector q from the centroid of the undisplaced set to the
        // node
        F2FScalar q[3];
        vec_diff(xs0bar, xs, q);

        // Get structural node displacement
        const F2FScalar *us = &Us[3 * indx];

        // Add contribution to aerodynamic surface node displacement
        addAdjDispContribution(W[j], r, Hinv, q, us, lam, rd, NULL, NULL);
      } else {
        indx = indx - ns;

        // Form the vector q from the centroid of the undisplaced set to the
        // node
        const F2FScalar *xs0 = &Xs[3 * indx];
        F2FScalar rxs0[3];
        memcpy(rxs0, xs0, 3 * sizeof(F2FScalar));
        rxs0[isymm] *= -1.0;

        F2FScalar q[3];
        vec_diff(xs0bar, rxs0, q);

        // Get structural node displacement
        const F2FScalar *us = &Us[3 * indx];

        // Add contribution to aerodynamic surface node displacement
        addAdjDispContribution(W[j], r, Hinv, q, us, lam, rd, NULL, NULL);
      }
    }

    F2FScalar *prod = &prods[3 * i];
    prod[0] -= rd[0];
    prod[1] -= rd[1];
    prod[2] -= rd[2];
  }
}

/*
  Apply the action of the displacement transfer w.r.t initial structural node
  locations Jacobian to the right of the transposed input vector

  Arguments
  ----------
  vecs  : input vector

  Returns
  --------
  prods : output vector
*/
void LinearizedMELD::applydDdxS0(const F2FScalar *vecs, F2FScalar *prods) {
  // Set products to zero
  F2FScalar *prods_global = new F2FScalar[3 * ns];
  memset(prods_global, 0.0, 3 * ns * sizeof(F2FScalar));

  for (int i = 0; i < na; i++) {
    // Get vector of adjoint variables and rotation matrix for each aerodynamic
    // node
    const F2FScalar *lam = &vecs[3 * i];

    // Point aerodynamic surface node location into a
    const F2FScalar *xa = &Xa[3 * i];

    // Compute the centroid of the initial set of nodes
    const F2FScalar *W = &global_W[i * nn];
    F2FScalar *xs0bar = &global_xs0bar[3 * i];

    // Compute the inverse of the point inertia matrix
    F2FScalar *H = &global_H[9 * i];
    F2FScalar Hinv[9];
    computePointInertiaInverse(H, Hinv);

    // Form the vector r from the initial centroid to the aerodynamic surface
    // node
    F2FScalar r[3];
    vec_diff(xs0bar, xa, r);

    // Zero the derivative contributions
    F2FScalar Hinvd[9], rd[3], xs0bard[3];
    memset(rd, 0, 3 * sizeof(F2FScalar));
    memset(xs0bard, 0, 3 * sizeof(F2FScalar));
    memset(Hinvd, 0, 9 * sizeof(F2FScalar));

    for (int j = 0; j < nn; j++) {
      // Get structural node location
      int indx = global_conn[nn * i + j];

      if (indx < ns) {
        F2FScalar *xs = &Xs[3 * indx];

        // Form the vector q from the centroid of the undisplaced set to the
        // node
        F2FScalar q[3];
        vec_diff(xs0bar, xs, q);  // q = xs - xs0bar

        // Get structural node displacement
        const F2FScalar *us = &Us[3 * indx];

        // Compute and add contribution to aerodynamic surface node
        F2FScalar qd[3];
        memset(qd, 0, 3 * sizeof(F2FScalar));
        addAdjDispContribution(W[j], r, Hinv, q, us, lam, rd, qd, Hinvd);

        // Add the contribution to the centroid
        xs0bard[0] -= qd[0];
        xs0bard[1] -= qd[1];
        xs0bard[2] -= qd[2];

        // Add the direct contribution to the point sensitivity
        F2FScalar *prod = &prods_global[3 * indx];
        prod[0] -= qd[0];
        prod[1] -= qd[1];
        prod[2] -= qd[2];
      } else {
        indx = indx - ns;

        // Form the vector q from the centroid of the undisplaced set to the
        // node
        const F2FScalar *xs0 = &Xs[3 * indx];
        F2FScalar rxs0[3];
        memcpy(rxs0, xs0, 3 * sizeof(F2FScalar));
        rxs0[isymm] *= -1.0;

        F2FScalar q[3];
        vec_diff(xs0bar, rxs0, q);  // q = rxs - xs0bar

        // Get structural node displacement
        const F2FScalar *us = &Us[3 * indx];

        // Compute the sensitivity to the node location
        F2FScalar qd[3];
        memset(qd, 0, 3 * sizeof(F2FScalar));
        addAdjDispContribution(W[j], r, Hinv, q, us, lam, rd, qd, Hinvd);

        // Add the contribution to the centroid
        xs0bard[0] -= qd[0];
        xs0bard[1] -= qd[1];
        xs0bard[2] -= qd[2];

        // Reflect the derivative
        qd[isymm] *= -1.0;

        F2FScalar *prod = &prods_global[3 * indx];
        prod[0] -= qd[0];
        prod[1] -= qd[1];
        prod[2] -= qd[2];
      }
    }

    // r = xa - xs0bar
    xs0bard[0] -= rd[0];
    xs0bard[1] -= rd[1];
    xs0bard[2] -= rd[2];

    // Now deal with the sensitivities w.r.t. Hinvd
    // Hinv = (H - I*tr(H))^{-1}
    F2FScalar Hd[9];
    adjPointInertiaInverse(Hinv, Hinvd, Hd);

    // Add the sensitivity due to the computation of H
    for (int j = 0; j < nn; j++) {
      int indx = global_conn[nn * i + j];

      if (indx < ns) {
        // Compute q = xs - xs0bar
        F2FScalar q[3];  // vector from centroid to node
        q[0] = Xs[3 * indx] - xs0bar[0];
        q[1] = Xs[3 * indx + 1] - xs0bar[1];
        q[2] = Xs[3 * indx + 2] - xs0bar[2];

        // Compute the derivative qd = w * (Hd + Hd^{T}) * q
        F2FScalar qd[3];
        qd[0] = W[j] * (2.0 * Hd[0] * q[0] + (Hd[1] + Hd[3]) * q[1] +
                        (Hd[2] + Hd[6]) * q[2]);
        qd[1] = W[j] * ((Hd[3] + Hd[1]) * q[0] + 2.0 * Hd[4] * q[1] +
                        (Hd[5] + Hd[7]) * q[2]);
        qd[2] = W[j] * ((Hd[6] + Hd[2]) * q[0] + (Hd[7] + Hd[5]) * q[1] +
                        2.0 * Hd[8] * q[2]);

        // Now apply the definition of q = xs - xs0bar
        F2FScalar *prod = &prods_global[3 * indx];
        prod[0] -= qd[0];
        prod[1] -= qd[1];
        prod[2] -= qd[2];

        xs0bard[0] -= qd[0];
        xs0bard[1] -= qd[1];
        xs0bard[2] -= qd[2];
      } else {
        indx = indx - ns;

        // Compute q = xs - xs0bar
        F2FScalar q[3];  // vector from centroid to node
        q[0] = Xs[3 * indx];
        q[1] = Xs[3 * indx + 1];
        q[2] = Xs[3 * indx + 2];

        // Reflect the point and subtract the centroid location
        q[isymm] *= -1.0;
        q[0] = q[0] - xs0bar[0];
        q[1] = q[1] - xs0bar[1];
        q[2] = q[2] - xs0bar[2];

        // Compute the derivative qd = w * (Hd + Hd^{T}) * q
        F2FScalar qd[3];
        qd[0] = W[j] * (2.0 * Hd[0] * q[0] + (Hd[1] + Hd[3]) * q[1] +
                        (Hd[2] + Hd[6]) * q[2]);
        qd[1] = W[j] * ((Hd[3] + Hd[1]) * q[0] + 2.0 * Hd[4] * q[1] +
                        (Hd[5] + Hd[7]) * q[2]);
        qd[2] = W[j] * ((Hd[6] + Hd[2]) * q[0] + (Hd[7] + Hd[5]) * q[1] +
                        2.0 * Hd[8] * q[2]);

        // Now apply the definition of q = xs - xs0bar
        xs0bard[0] -= qd[0];
        xs0bard[1] -= qd[1];
        xs0bard[2] -= qd[2];

        qd[isymm] *= -1.0;

        // Add the contribution to the poin
        F2FScalar *prod = &prods_global[3 * indx];
        prod[0] -= qd[0];
        prod[1] -= qd[1];
        prod[2] -= qd[2];
      }
    }

    F2FScalar rxs0bard[3];
    rxs0bard[0] = xs0bard[0];
    rxs0bard[1] = xs0bard[1];
    rxs0bard[2] = xs0bard[2];
    if (isymm >= 0) {
      rxs0bard[isymm] *= -1.0;
    }

    // Add the contribution from the centroid
    for (int j = 0; j < nn; j++) {
      int indx = global_conn[nn * i + j];

      if (indx < ns) {
        F2FScalar *prod = &prods_global[3 * indx];
        prod[0] -= W[j] * xs0bard[0];
        prod[1] -= W[j] * xs0bard[1];
        prod[2] -= W[j] * xs0bard[2];
      } else {
        indx = indx - ns;
        F2FScalar *prod = &prods_global[3 * indx];
        prod[0] -= W[j] * rxs0bard[0];
        prod[1] -= W[j] * rxs0bard[1];
        prod[2] -= W[j] * rxs0bard[2];
      }
    }
  }

  // distribute the results to the structural processors
  structAddScatter(3 * ns, prods_global, 3 * ns_local, prods);

  // clean up allocated memory
  delete[] prods_global;
}

/*
  Apply the action of the load transfer w.r.t initial aerodynamic surface node
  locations Jacobian to the right of the transposed input vector

  Arguments
  ----------
  vecs  : input vector

  Returns
  --------
  prods : output vector
*/
void LinearizedMELD::applydLdxA0(const F2FScalar *vecs, F2FScalar *prods) {
  F2FScalar *vecs_global = new F2FScalar[3 * ns];
  structGatherBcast(3 * ns_local, vecs, 3 * ns, vecs_global);

  // Zero products
  memset(prods, 0, 3 * na * sizeof(F2FScalar));

  // Loop over aerodynamic surface nodes
  for (int i = 0; i < na; i++) {
    // Get aerodynamic surface node location and load
    F2FScalar *a = &Xa[3 * i];
    const F2FScalar *fa = &Fa[3 * i];

    // Compute vector d from centroid to aerodynamic surface node
    F2FScalar *xs0bar = &global_xs0bar[3 * i];
    F2FScalar r[3];
    vec_diff(xs0bar, a, r);

    // Compute inverse of point inertia matrix
    F2FScalar Hinv[9];
    computePointInertiaInverse(&global_H[9 * i], Hinv);

    // Zero the derivative contributions
    F2FScalar rd[3];
    memset(rd, 0, 3 * sizeof(F2FScalar));

    // Loop over linked structural nodes
    for (int j = 0; j < nn; j++) {
      // Get structural node using index from connectivity
      int indx = global_conn[nn * i + j];

      if (indx < ns) {
        F2FScalar *xs = &Xs[3 * indx];

        // Compute vector q from centroid to structural node
        F2FScalar q[3];
        vec_diff(xs0bar, xs, q);

        // Compute load contribution
        const F2FScalar *fsd = &vecs_global[3 * indx];
        F2FScalar w = global_W[nn * i + j];

        addAdjLoadContribution(w, r, Hinv, q, fa, fsd, rd, NULL, NULL);
      } else {
        indx = indx - ns;

        // Compute vector q from centroid to structural node using the
        // symmetry condition
        const F2FScalar *xs0 = &Xs[3 * indx];
        F2FScalar rxs0[3];
        memcpy(rxs0, xs0, 3 * sizeof(F2FScalar));
        rxs0[isymm] *= -1.0;

        // Compute vector q from centroid to structural node
        F2FScalar q[3];
        vec_diff(xs0bar, rxs0, q);

        // Compute load contribution
        const F2FScalar *fsd = &vecs_global[3 * indx];
        F2FScalar w = global_W[nn * i + j];

        addAdjLoadContribution(w, r, Hinv, q, fa, fsd, rd, NULL, NULL);
      }
    }

    F2FScalar *prod = &prods[3 * i];
    prod[0] -= rd[0];
    prod[1] -= rd[1];
    prod[2] -= rd[2];
  }
}

/*
  Apply the action of the load transfer w.r.t initial structural node locations
  Jacobian to the right of the transposed input vector

  Arguments
  ----------
  vecs  : input vector

  Returns
  --------
  prods : output vector
*/
void LinearizedMELD::applydLdxS0(const F2FScalar *vecs, F2FScalar *prods) {
  F2FScalar *vecs_global = new F2FScalar[3 * ns];
  structGatherBcast(3 * ns_local, vecs, 3 * ns, vecs_global);

  // Set products to zero
  F2FScalar *prods_global = new F2FScalar[3 * ns];
  memset(prods_global, 0.0, 3 * ns * sizeof(F2FScalar));

  // Loop over aerodynamic surface nodes
  for (int i = 0; i < na; i++) {
    // Get aerodynamic surface node location and load
    F2FScalar *a = &Xa[3 * i];
    const F2FScalar *fa = &Fa[3 * i];

    // Compute vector d from centroid to aerodynamic surface node
    const F2FScalar *W = &global_W[i * nn];
    F2FScalar *xs0bar = &global_xs0bar[3 * i];
    F2FScalar r[3];
    vec_diff(xs0bar, a, r);

    // Compute inverse of point inertia matrix
    F2FScalar Hinv[9];
    computePointInertiaInverse(&global_H[9 * i], Hinv);

    // Zero the derivative contributions
    F2FScalar Hinvd[9], rd[3], xs0bard[3];
    memset(rd, 0, 3 * sizeof(F2FScalar));
    memset(xs0bard, 0, 3 * sizeof(F2FScalar));
    memset(Hinvd, 0, 9 * sizeof(F2FScalar));

    // Loop over linked structural nodes
    for (int j = 0; j < nn; j++) {
      // Get structural node using index from connectivity
      int indx = global_conn[nn * i + j];

      if (indx < ns) {
        F2FScalar *xs = &Xs[3 * indx];

        // Compute vector q from centroid to structural node
        F2FScalar q[3];
        vec_diff(xs0bar, xs, q);

        // Get the load adjoint
        const F2FScalar *fsd = &vecs_global[3 * indx];

        // Compute load contribution to the adjoint
        F2FScalar qd[3];
        memset(qd, 0, 3 * sizeof(F2FScalar));
        addAdjLoadContribution(W[j], r, Hinv, q, fa, fsd, rd, qd, Hinvd);

        // Add the contribution to the centroid
        xs0bard[0] -= qd[0];
        xs0bard[1] -= qd[1];
        xs0bard[2] -= qd[2];

        // Add the direct contribution to the point sensitivity
        F2FScalar *prod = &prods_global[3 * indx];
        prod[0] -= qd[0];
        prod[1] -= qd[1];
        prod[2] -= qd[2];
      } else {
        indx = indx - ns;

        // Form the vector q from the centroid of the undisplaced set to the
        // node
        const F2FScalar *xs0 = &Xs[3 * indx];
        F2FScalar rxs0[3];
        memcpy(rxs0, xs0, 3 * sizeof(F2FScalar));
        rxs0[isymm] *= -1.0;

        F2FScalar q[3];
        vec_diff(xs0bar, rxs0, q);  // q = rxs - xs0bar

        // Get the load adjoint
        const F2FScalar *fsd = &vecs_global[3 * indx];

        // Compute load contribution to the adjoint
        F2FScalar qd[3];
        memset(qd, 0, 3 * sizeof(F2FScalar));
        addAdjLoadContribution(W[j], r, Hinv, q, fa, fsd, rd, qd, Hinvd);

        // Add the contribution to the centroid
        xs0bard[0] -= qd[0];
        xs0bard[1] -= qd[1];
        xs0bard[2] -= qd[2];

        // Reflect the derivative
        qd[isymm] *= -1.0;

        // Add the direct contribution to the point sensitivity
        F2FScalar *prod = &prods_global[3 * indx];
        prod[0] -= qd[0];
        prod[1] -= qd[1];
        prod[2] -= qd[2];
      }
    }

    // r = xa - xs0bar
    xs0bard[0] -= rd[0];
    xs0bard[1] -= rd[1];
    xs0bard[2] -= rd[2];

    // Now deal with the sensitivities w.r.t. Hinvd
    // Hinv = (H - I*tr(H))^{-1}
    F2FScalar Hd[9];
    adjPointInertiaInverse(Hinv, Hinvd, Hd);

    // Add the sensitivity due to the computation of H
    for (int j = 0; j < nn; j++) {
      int indx = global_conn[nn * i + j];

      if (indx < ns) {
        // Compute q = xs - xs0bar
        F2FScalar q[3];  // vector from centroid to node
        q[0] = Xs[3 * indx] - xs0bar[0];
        q[1] = Xs[3 * indx + 1] - xs0bar[1];
        q[2] = Xs[3 * indx + 2] - xs0bar[2];

        // Compute the derivative qd = w * (Hd + Hd^{T}) * q
        F2FScalar qd[3];
        qd[0] = W[j] * (2.0 * Hd[0] * q[0] + (Hd[1] + Hd[3]) * q[1] +
                        (Hd[2] + Hd[6]) * q[2]);
        qd[1] = W[j] * ((Hd[3] + Hd[1]) * q[0] + 2.0 * Hd[4] * q[1] +
                        (Hd[5] + Hd[7]) * q[2]);
        qd[2] = W[j] * ((Hd[6] + Hd[2]) * q[0] + (Hd[7] + Hd[5]) * q[1] +
                        2.0 * Hd[8] * q[2]);

        // Now apply the definition of q = xs - xs0bar

        // Add the contribution to the poin
        F2FScalar *prod = &prods_global[3 * indx];
        prod[0] -= qd[0];
        prod[1] -= qd[1];
        prod[2] -= qd[2];

        xs0bard[0] -= qd[0];
        xs0bard[1] -= qd[1];
        xs0bard[2] -= qd[2];
      } else {
        indx = indx - ns;

        // Compute q = xs - xs0bar
        F2FScalar q[3];  // vector from centroid to node
        q[0] = Xs[3 * indx];
        q[1] = Xs[3 * indx + 1];
        q[2] = Xs[3 * indx + 2];

        // Reflect the point and subtract the centroid location
        q[isymm] *= -1.0;
        q[0] = q[0] - xs0bar[0];
        q[1] = q[1] - xs0bar[1];
        q[2] = q[2] - xs0bar[2];

        // Compute the derivative qd = w * (Hd + Hd^{T}) * q
        F2FScalar qd[3];
        qd[0] = W[j] * (2.0 * Hd[0] * q[0] + (Hd[1] + Hd[3]) * q[1] +
                        (Hd[2] + Hd[6]) * q[2]);
        qd[1] = W[j] * ((Hd[3] + Hd[1]) * q[0] + 2.0 * Hd[4] * q[1] +
                        (Hd[5] + Hd[7]) * q[2]);
        qd[2] = W[j] * ((Hd[6] + Hd[2]) * q[0] + (Hd[7] + Hd[5]) * q[1] +
                        2.0 * Hd[8] * q[2]);

        // Now apply the definition of q = xs - xs0bar
        xs0bard[0] -= qd[0];
        xs0bard[1] -= qd[1];
        xs0bard[2] -= qd[2];

        qd[isymm] *= -1.0;

        // Add the contribution to the poin
        F2FScalar *prod = &prods_global[3 * indx];
        prod[0] -= qd[0];
        prod[1] -= qd[1];
        prod[2] -= qd[2];
      }
    }

    F2FScalar rxs0bard[3];
    rxs0bard[0] = xs0bard[0];
    rxs0bard[1] = xs0bard[1];
    rxs0bard[2] = xs0bard[2];
    if (isymm >= 0) {
      rxs0bard[isymm] *= -1.0;
    }

    // Add the contribution from the centroid
    for (int j = 0; j < nn; j++) {
      int indx = global_conn[nn * i + j];

      if (indx < ns) {
        F2FScalar *prod = &prods_global[3 * indx];
        prod[0] -= W[j] * xs0bard[0];
        prod[1] -= W[j] * xs0bard[1];
        prod[2] -= W[j] * xs0bard[2];
      } else {
        indx = indx - ns;
        F2FScalar *prod = &prods_global[3 * indx];
        prod[0] -= W[j] * rxs0bard[0];
        prod[1] -= W[j] * rxs0bard[1];
        prod[2] -= W[j] * rxs0bard[2];
      }
    }
  }

  // distribute the results to the structural processors
  structAddScatter(3 * ns, prods_global, 3 * ns_local, prods);

  // clean up allocated memory
  delete[] prods_global;
  delete[] vecs_global;
}
