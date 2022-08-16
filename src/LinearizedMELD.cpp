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
  printf("Transfer scheme [%i]: Creating scheme of type LinearizedMELD...\n",
         object_id);
}

LinearizedMELD::~LinearizedMELD() {
  // Free the data for the transfers
  if (global_H) {
    delete[] global_H;
  }

  printf("Transfer scheme [%i]: freeing LinearizedMELD data...\n", object_id);
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

  // Create aerostructural connectivity
  global_conn = new int[nn * na];
  setAeroStructConn(global_conn);

  // Allocate and compute the weights
  global_W = new F2FScalar[nn * na];
  computeWeights(global_W);

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
  // Copy prescribed displacements into displacement vector
  memcpy(Us, struct_disps, 3 * ns * sizeof(F2FScalar));

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
      F2FScalar *xs = &Xs[3 * indx];

      // Form the vector q from the centroid of the undisplaced set to the node
      F2FScalar q[3];
      vec_diff(xs0bar, xs, q);

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
  // Compute matrix = w*(qx*Hinv*dx + I)
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
  memset(struct_loads, 0, 3 * ns * sizeof(F2FScalar));

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
      F2FScalar *xs = &Xs[3 * indx];

      // Compute vector q from centroid to structural node
      F2FScalar q[3];
      vec_diff(xs0bar, xs, q);

      // Compute load contribution
      F2FScalar *fs = &struct_loads[3 * indx];
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
  for (int j = 0; j < 3 * ns; j++) {
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
  memset(prods, 0, 3 * ns * sizeof(F2FScalar));
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
  memset(prods, 0, 3 * ns * sizeof(F2FScalar));
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
  memset(prods, 0, 3 * na * sizeof(F2FScalar));
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
  memset(prods, 0, 3 * ns * sizeof(F2FScalar));
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
  memset(prods, 0, 3 * na * sizeof(F2FScalar));
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
  memset(prods, 0, 3 * ns * sizeof(F2FScalar));
}
