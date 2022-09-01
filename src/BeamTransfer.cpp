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
#include "BeamTransfer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstring>

#include "LocatePoint.h"

/*
  Create the beam transfer load and displacement transfer object. This
  assumes that all information stored in the transfer object.
*/
BeamTransfer::BeamTransfer(MPI_Comm global_comm, MPI_Comm struct_comm,
                           int struct_root, MPI_Comm aero_comm, int aero_root,
                           const int *conn_, int nelems, int order,
                           int dof_per_node)
    : LDTransferScheme(global_comm, struct_comm, struct_root, aero_comm,
                       aero_root, dof_per_node),
      nelems(nelems),
      order(order),
      dof_per_node(dof_per_node) {
  int rank;
  MPI_Comm_rank(global_comm, &rank);
  if (!(order == 2 || order == 3)) {
    if (rank == struct_root) {
      printf(
          "Transfer scheme [%i]: BeamTransfer only uses order = 2 || order = "
          "3\n",
          object_id);
    }
  }

  // Initialize the local connectivity information
  aero_pt_to_elem = NULL;
  aero_pt_to_param = NULL;

  // Create the connectivity
  conn = new int[order * nelems];
  memcpy(conn, conn_, order * nelems * sizeof(int));

  // Notify user of the type of transfer scheme they are using
  if (rank == struct_root) {
    printf("Transfer scheme [%i]: Creating scheme of type BeamTransfer...\n",
           object_id);
  }
}

BeamTransfer::~BeamTransfer() {
  delete[] conn;
  if (aero_pt_to_elem) {
    delete[] aero_pt_to_elem;
  }
  if (aero_pt_to_param) {
    delete[] aero_pt_to_param;
  }

  int rank;
  MPI_Comm_rank(global_comm, &rank);
  if (rank == struct_root) {
    printf("Transfer scheme [%i]: freeing BeamTransfer data...\n", object_id);
  }
}

/*
  Initialize the number of points
*/
void BeamTransfer::initialize() {
  // global number of structural nodes
  distributeStructuralMesh();

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
    Us = new F2FScalar[dof_per_node * ns];
    memset(Us, 0, dof_per_node * ns * sizeof(F2FScalar));
  }

  // Set up the node to element data
  int *node_to_beam_ptr = new int[ns + 1];
  memset(node_to_beam_ptr, 0, (ns + 1) * sizeof(int));

  // Count up all the times each element refers to each node
  for (int i = 0; i < order * nelems; i++) {
    node_to_beam_ptr[conn[i] + 1]++;
  }

  // Reset the array so that it will point in to the list
  // of elements
  for (int i = 0; i < ns; i++) {
    node_to_beam_ptr[i + 1] += node_to_beam_ptr[i];
  }

  // Allocate the array of nodes
  int *node_to_beam = new int[node_to_beam_ptr[ns]];

  for (int i = 0; i < nelems; i++) {
    // Set the reference to the i-th element
    for (int j = 0; j < order; j++) {
      node_to_beam[node_to_beam_ptr[conn[order * i + j]]] = i;
      node_to_beam_ptr[conn[order * i + j]]++;
    }
  }

  // Reset the node->beam pointer array
  for (int i = ns; i > 0; i--) {
    node_to_beam_ptr[i] = node_to_beam_ptr[i - 1];
  }
  node_to_beam_ptr[0] = 0;

  // Create instance of LocatePoint class to perform the following
  // searches
  int min_bin_size = 10;
  LocatePoint *locator = new LocatePoint(Xs, ns, min_bin_size);

  // Allocate space for the data
  aero_pt_to_elem = new int[na];
  aero_pt_to_param = new double[na];

  // Find the node that is closest to the beam
  for (int i = 0; i < na; i++) {
    // Locate the structural node that is closest to the beam
    int node = locator->locateClosest(&Xa[3 * i]);

    // Find the minimum distance from any of the adjacent elements
    F2FScalar min_dist = 1e20;
    int min_elem = 0;
    double min_xi = 0.0;

    for (int j = node_to_beam_ptr[node]; j < node_to_beam_ptr[node + 1]; j++) {
      int elem = node_to_beam[j];

      double xi;
      F2FScalar dist = findParametricPoint(
          &Xs[3 * conn[order * elem]], &Xs[3 * conn[order * (elem + 1) - 1]],
          &Xa[3 * i], &xi);
      if (F2FRealPart(dist) < F2FRealPart(min_dist)) {
        min_dist = dist;
        min_elem = elem;
        min_xi = xi;
      }
    }

    // Record the closest parametric point for each aerodynamic
    // point in the mesh
    aero_pt_to_elem[i] = min_elem;
    aero_pt_to_param[i] = min_xi;
  }

  delete[] node_to_beam_ptr;
  delete[] node_to_beam;
}

/*
  Find the closest parametric point to the line
*/
F2FScalar BeamTransfer::findParametricPoint(const F2FScalar X1[],
                                            const F2FScalar X2[],
                                            const F2FScalar Xa[], double *xi) {
  // Compute the vector between the two points
  F2FScalar d[3];
  d[0] = X2[0] - X1[0];
  d[1] = X2[1] - X1[1];
  d[2] = X2[2] - X1[2];

  // Compute the distance between the two points
  F2FScalar L = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
  F2FScalar Linv = 0.0;
  if (F2FRealPart(L) > 0.0) {
    Linv = 1.0 / L;
  }

  // Compute the normalized vector
  d[0] = Linv * d[0];
  d[1] = Linv * d[1];
  d[2] = Linv * d[2];

  // Compute the projection on to the line
  F2FScalar proj =
      d[0] * (Xa[0] - X1[0]) + d[1] * (Xa[1] - X1[1]) + d[2] * (Xa[2] - X1[2]);

  if (F2FRealPart(proj) < 0.0) {
    *xi = -1.0;
    return ((Xa[0] - X1[0]) * (Xa[0] - X1[0]) +
            (Xa[1] - X1[1]) * (Xa[1] - X1[1]) +
            (Xa[2] - X1[2]) * (Xa[2] - X1[2]));
  } else if (F2FRealPart(proj) > F2FRealPart(L)) {
    *xi = 1.0;
    return ((Xa[0] - X2[0]) * (Xa[0] - X2[0]) +
            (Xa[1] - X2[1]) * (Xa[1] - X2[1]) +
            (Xa[2] - X2[2]) * (Xa[2] - X2[2]));
  }

  // Based on the projection value, compute the parametric location
  *xi = -1.0 + 2.0 * F2FRealPart(proj / L);

  // Compute the weight
  F2FScalar u = 0.5 * (1.0 - *xi);

  F2FScalar X[3];
  X[0] = u * X1[0] + (1.0 - u) * X2[0];
  X[1] = u * X1[1] + (1.0 - u) * X2[1];
  X[2] = u * X1[2] + (1.0 - u) * X2[2];

  return ((Xa[0] - X[0]) * (Xa[0] - X[0]) + (Xa[1] - X[1]) * (Xa[1] - X[1]) +
          (Xa[2] - X[2]) * (Xa[2] - X[2]));
}

/*
  Compute the rotational part of the load and displacement transfer.

  The input d is the vector between the aerodynamic node and its
  corresponding structural node location, such that d = Xa - Xs.  The
  output r is computed as: r = (C^{T} - I)*d which provides the
  displacement contribution due to the rigid link movement.
*/
void BeamTransfer::computeRotation(const F2FScalar *q, const F2FScalar *d,
                                   F2FScalar r[]) {
  // Compute R = (C - I)
  F2FScalar R[9];
  if (dof_per_node == 6) {
    // R = q^{x}
    R[0] = 0.0;
    R[1] = -q[2];
    R[2] = q[1];

    R[3] = q[2];
    R[4] = 0.0;
    R[5] = -q[0];

    R[6] = -q[1];
    R[7] = q[0];
    R[8] = 0.0;
  } else {
    R[0] = -2.0 * (q[2] * q[2] + q[3] * q[3]);
    R[1] = 2.0 * (q[1] * q[2] + q[3] * q[0]);
    R[2] = 2.0 * (q[1] * q[3] - q[2] * q[0]);

    R[3] = 2.0 * (q[2] * q[1] - q[3] * q[0]);
    R[4] = -2.0 * (q[1] * q[1] + q[3] * q[3]);
    R[5] = 2.0 * (q[2] * q[3] + q[1] * q[0]);

    R[6] = 2.0 * (q[3] * q[1] + q[2] * q[0]);
    R[7] = 2.0 * (q[3] * q[2] - q[1] * q[0]);
    R[8] = -2.0 * (q[1] * q[1] + q[2] * q[2]);
  }

  // Compute r = R^{T}*d
  r[0] = R[0] * d[0] + R[3] * d[1] + R[6] * d[2];
  r[1] = R[1] * d[0] + R[4] * d[1] + R[7] * d[2];
  r[2] = R[2] * d[0] + R[5] * d[1] + R[8] * d[2];
}

/*
  Compute the transpose product of the rotation
*/
void BeamTransfer::computeRotationTranspose(const F2FScalar *q,
                                            const F2FScalar *d, F2FScalar r[]) {
  // Compute R = (C - I)
  F2FScalar R[9];
  if (dof_per_node == 6) {
    // R = q^{x}
    R[0] = 0.0;
    R[1] = -q[2];
    R[2] = q[1];

    R[3] = q[2];
    R[4] = 0.0;
    R[5] = -q[0];

    R[6] = -q[1];
    R[7] = q[0];
    R[8] = 0.0;
  } else {
    R[0] = -2.0 * (q[2] * q[2] + q[3] * q[3]);
    R[1] = 2.0 * (q[1] * q[2] + q[3] * q[0]);
    R[2] = 2.0 * (q[1] * q[3] - q[2] * q[0]);

    R[3] = 2.0 * (q[2] * q[1] - q[3] * q[0]);
    R[4] = -2.0 * (q[1] * q[1] + q[3] * q[3]);
    R[5] = 2.0 * (q[2] * q[3] + q[1] * q[0]);

    R[6] = 2.0 * (q[3] * q[1] + q[2] * q[0]);
    R[7] = 2.0 * (q[3] * q[2] - q[1] * q[0]);
    R[8] = -2.0 * (q[1] * q[1] + q[2] * q[2]);
  }

  // Compute r = R*d
  r[0] = R[0] * d[0] + R[1] * d[1] + R[2] * d[2];
  r[1] = R[3] * d[0] + R[4] * d[1] + R[5] * d[2];
  r[2] = R[6] * d[0] + R[7] * d[1] + R[8] * d[2];
}

/*
  Compute the derivative of
*/
void BeamTransfer::computeRotationDerivProduct(const F2FScalar *v,
                                               const F2FScalar *q,
                                               const F2FScalar *d,
                                               F2FScalar r[]) {
  // Compute R = (C - I)
  F2FScalar R[9];
  if (dof_per_node == 6) {
    // R = v^{x}
    R[0] = 0.0;
    R[1] = -v[2];
    R[2] = v[1];

    R[3] = v[2];
    R[4] = 0.0;
    R[5] = -v[0];

    R[6] = -v[1];
    R[7] = v[0];
    R[8] = 0.0;
  } else {
    R[0] = -4.0 * (q[2] * v[2] + q[3] * v[3]);
    R[1] = 2.0 * (q[1] * v[2] + v[1] * q[2] + q[3] * v[0] + v[3] * q[0]);
    R[2] = 2.0 * (q[1] * v[3] + v[1] * q[3] - q[2] * v[0] - v[2] * q[0]);

    R[3] = 2.0 * (q[2] * v[1] + v[2] * q[1] - q[3] * v[0] - v[3] * q[0]);
    R[4] = -4.0 * (q[1] * v[1] + q[3] * v[3]);
    R[5] = 2.0 * (q[2] * v[3] + v[2] * q[3] + q[1] * v[0] + v[1] * q[0]);

    R[6] = 2.0 * (q[3] * v[1] + v[3] * q[1] + q[2] * v[0] + v[2] * q[0]);
    R[7] = 2.0 * (q[3] * v[2] + v[3] * q[2] - q[1] * v[0] - v[1] * q[0]);
    R[8] = -4.0 * (q[1] * v[1] + q[2] * v[2]);
  }

  // Compute r = R^{T}*d
  r[0] = R[0] * d[0] + R[3] * d[1] + R[6] * d[2];
  r[1] = R[1] * d[0] + R[4] * d[1] + R[7] * d[2];
  r[2] = R[2] * d[0] + R[5] * d[1] + R[8] * d[2];
}

/*
  Add the derivative of the product of s*fa^{T}*(C^{T} - I)*d with
  respect to the variables q to the output vector fs.

  In other words, compute this:

  fs += s*fa^{T}*d(C^{T}*d)/dq
*/
void BeamTransfer::addTransposeRotationDeriv(const double scale,
                                             const F2FScalar *q,
                                             const F2FScalar *d,
                                             const F2FScalar *fa,
                                             F2FScalar fs[]) {
  F2FScalar R[9], r[3];

  if (dof_per_node == 6) {
    // Compute the derivative w.r.t. q[0]
    R[0] = 0.0;
    R[1] = 0.0;
    R[2] = 0.0;

    R[3] = 0.0;
    R[4] = 0.0;
    R[5] = -1.0;

    R[6] = 0.0;
    R[7] = 1.0;
    R[8] = 0.0;

    r[0] = R[0] * d[0] + R[3] * d[1] + R[6] * d[2];
    r[1] = R[1] * d[0] + R[4] * d[1] + R[7] * d[2];
    r[2] = R[2] * d[0] + R[5] * d[1] + R[8] * d[2];

    fs[0] += scale * (r[0] * fa[0] + r[1] * fa[1] + r[2] * fa[2]);

    // Derivative w.r.t. q[1]
    R[0] = 0.0;
    R[1] = 0.0;
    R[2] = 1.0;

    R[3] = 0.0;
    R[4] = 0.0;
    R[5] = 0.0;

    R[6] = -1.0;
    R[7] = 0.0;
    R[8] = 0.0;

    r[0] = R[0] * d[0] + R[3] * d[1] + R[6] * d[2];
    r[1] = R[1] * d[0] + R[4] * d[1] + R[7] * d[2];
    r[2] = R[2] * d[0] + R[5] * d[1] + R[8] * d[2];

    fs[1] += scale * (r[0] * fa[0] + r[1] * fa[1] + r[2] * fa[2]);

    // Derivative w.r.t. q[2]
    R[0] = 0.0;
    R[1] = -1.0;
    R[2] = 0.0;

    R[3] = 1.0;
    R[4] = 0.0;
    R[5] = 0.0;

    R[6] = 0.0;
    R[7] = 0.0;
    R[8] = 0.0;

    r[0] = R[0] * d[0] + R[3] * d[1] + R[6] * d[2];
    r[1] = R[1] * d[0] + R[4] * d[1] + R[7] * d[2];
    r[2] = R[2] * d[0] + R[5] * d[1] + R[8] * d[2];

    fs[2] += scale * (r[0] * fa[0] + r[1] * fa[1] + r[2] * fa[2]);
  } else {
    // Compute the derivative w.r.t. q[0]
    R[0] = 0.0;
    R[1] = 2.0 * q[3];
    R[2] = -2.0 * q[2];

    R[3] = -2.0 * q[3];
    R[4] = 0.0;
    R[5] = 2.0 * q[1];

    R[6] = 2.0 * q[2];
    R[7] = -2.0 * q[1];
    R[8] = 0.0;

    r[0] = R[0] * d[0] + R[3] * d[1] + R[6] * d[2];
    r[1] = R[1] * d[0] + R[4] * d[1] + R[7] * d[2];
    r[2] = R[2] * d[0] + R[5] * d[1] + R[8] * d[2];

    fs[0] += scale * (r[0] * fa[0] + r[1] * fa[1] + r[2] * fa[2]);

    // Derivative w.r.t. q[1]
    R[0] = 0.0;
    R[1] = 2.0 * q[2];
    R[2] = 2.0 * q[3];

    R[3] = 2.0 * q[2];
    R[4] = -4.0 * q[1];
    R[5] = 2.0 * q[0];

    R[6] = 2.0 * q[3];
    R[7] = -2.0 * q[0];
    R[8] = -4.0 * q[1];

    r[0] = R[0] * d[0] + R[3] * d[1] + R[6] * d[2];
    r[1] = R[1] * d[0] + R[4] * d[1] + R[7] * d[2];
    r[2] = R[2] * d[0] + R[5] * d[1] + R[8] * d[2];

    fs[1] += scale * (r[0] * fa[0] + r[1] * fa[1] + r[2] * fa[2]);

    // Derivative w.r.t. q[2]
    R[0] = -4.0 * q[2];
    R[1] = 2.0 * q[1];
    R[2] = -2.0 * q[0];

    R[3] = 2.0 * q[1];
    R[4] = 0.0;
    R[5] = 2.0 * q[3];

    R[6] = 2.0 * q[0];
    R[7] = 2.0 * q[3];
    R[8] = -4.0 * q[2];

    r[0] = R[0] * d[0] + R[3] * d[1] + R[6] * d[2];
    r[1] = R[1] * d[0] + R[4] * d[1] + R[7] * d[2];
    r[2] = R[2] * d[0] + R[5] * d[1] + R[8] * d[2];

    fs[2] += scale * (r[0] * fa[0] + r[1] * fa[1] + r[2] * fa[2]);

    // Derivative w.r.t. q[3]
    R[0] = -4.0 * q[3];
    R[1] = 2.0 * q[0];
    R[2] = 2.0 * q[1];

    R[3] = -2.0 * q[0];
    R[4] = -4.0 * q[3];
    R[5] = 2.0 * q[2];

    R[6] = 2.0 * q[1];
    R[7] = 2.0 * q[2];
    R[8] = 0.0;

    r[0] = R[0] * d[0] + R[3] * d[1] + R[6] * d[2];
    r[1] = R[1] * d[0] + R[4] * d[1] + R[7] * d[2];
    r[2] = R[2] * d[0] + R[5] * d[1] + R[8] * d[2];

    fs[3] += scale * (r[0] * fa[0] + r[1] * fa[1] + r[2] * fa[2]);
  }
}

void BeamTransfer::addTransposeRotationDerivAdjoint(const double scale,
                                                    const F2FScalar *q,
                                                    const F2FScalar *fa,
                                                    const F2FScalar *fs,
                                                    F2FScalar psi[]) {
  F2FScalar R[9];

  if (dof_per_node == 6) {
    // Compute the derivative w.r.t. q[0]
    R[0] = 0.0;
    R[1] = 0.0;
    R[2] = 0.0;

    R[3] = 0.0;
    R[4] = 0.0;
    R[5] = -1.0;

    R[6] = 0.0;
    R[7] = 1.0;
    R[8] = 0.0;

    // fs = fa^{T}*[d(R^{T})/dq1]*d
    psi[0] += scale * fs[0] * (R[0] * fa[0] + R[1] * fa[1] + R[2] * fa[2]);
    psi[1] += scale * fs[0] * (R[3] * fa[0] + R[4] * fa[1] + R[5] * fa[2]);
    psi[2] += scale * fs[0] * (R[6] * fa[0] + R[7] * fa[1] + R[8] * fa[2]);

    // Derivative w.r.t. q[1]
    R[0] = 0.0;
    R[1] = 0.0;
    R[2] = 1.0;

    R[3] = 0.0;
    R[4] = 0.0;
    R[5] = 0.0;

    R[6] = -1.0;
    R[7] = 0.0;
    R[8] = 0.0;

    psi[0] += scale * fs[1] * (R[0] * fa[0] + R[1] * fa[1] + R[2] * fa[2]);
    psi[1] += scale * fs[1] * (R[3] * fa[0] + R[4] * fa[1] + R[5] * fa[2]);
    psi[2] += scale * fs[1] * (R[6] * fa[0] + R[7] * fa[1] + R[8] * fa[2]);

    // Derivative w.r.t. q[2]
    R[0] = 0.0;
    R[1] = -1.0;
    R[2] = 0.0;

    R[3] = 1.0;
    R[4] = 0.0;
    R[5] = 0.0;

    R[6] = 0.0;
    R[7] = 0.0;
    R[8] = 0.0;

    psi[0] += scale * fs[2] * (R[0] * fa[0] + R[1] * fa[1] + R[2] * fa[2]);
    psi[1] += scale * fs[2] * (R[3] * fa[0] + R[4] * fa[1] + R[5] * fa[2]);
    psi[2] += scale * fs[2] * (R[6] * fa[0] + R[7] * fa[1] + R[8] * fa[2]);

  } else {
    // Compute the derivative w.r.t. q[0]
    R[0] = 0.0;
    R[1] = 2.0 * q[3];
    R[2] = -2.0 * q[2];

    R[3] = -2.0 * q[3];
    R[4] = 0.0;
    R[5] = 2.0 * q[1];

    R[6] = 2.0 * q[2];
    R[7] = -2.0 * q[1];
    R[8] = 0.0;

    // fs = fa^{T}*[d(R^{T})/dq1]*d
    psi[0] += scale * fs[0] * (R[0] * fa[0] + R[1] * fa[1] + R[2] * fa[2]);
    psi[1] += scale * fs[0] * (R[3] * fa[0] + R[4] * fa[1] + R[5] * fa[2]);
    psi[2] += scale * fs[0] * (R[6] * fa[0] + R[7] * fa[1] + R[8] * fa[2]);

    // Derivative w.r.t. q[1]
    R[0] = 0.0;
    R[1] = 2.0 * q[2];
    R[2] = 2.0 * q[3];

    R[3] = 2.0 * q[2];
    R[4] = -4.0 * q[1];
    R[5] = 2.0 * q[0];

    R[6] = 2.0 * q[3];
    R[7] = -2.0 * q[0];
    R[8] = -4.0 * q[1];

    psi[0] += scale * fs[1] * (R[0] * fa[0] + R[1] * fa[1] + R[2] * fa[2]);
    psi[1] += scale * fs[1] * (R[3] * fa[0] + R[4] * fa[1] + R[5] * fa[2]);
    psi[2] += scale * fs[1] * (R[6] * fa[0] + R[7] * fa[1] + R[8] * fa[2]);

    // Derivative w.r.t. q[2]
    R[0] = -4.0 * q[2];
    R[1] = 2.0 * q[1];
    R[2] = -2.0 * q[0];

    R[3] = 2.0 * q[1];
    R[4] = 0.0;
    R[5] = 2.0 * q[3];

    R[6] = 2.0 * q[0];
    R[7] = 2.0 * q[3];
    R[8] = -4.0 * q[2];

    psi[0] += scale * fs[2] * (R[0] * fa[0] + R[1] * fa[1] + R[2] * fa[2]);
    psi[1] += scale * fs[2] * (R[3] * fa[0] + R[4] * fa[1] + R[5] * fa[2]);
    psi[2] += scale * fs[2] * (R[6] * fa[0] + R[7] * fa[1] + R[8] * fa[2]);

    // Derivative w.r.t. q[3]
    R[0] = -4.0 * q[3];
    R[1] = 2.0 * q[0];
    R[2] = 2.0 * q[1];

    R[3] = -2.0 * q[0];
    R[4] = -4.0 * q[3];
    R[5] = 2.0 * q[2];

    R[6] = 2.0 * q[1];
    R[7] = 2.0 * q[2];
    R[8] = 0.0;

    psi[0] += scale * fs[3] * (R[0] * fa[0] + R[1] * fa[1] + R[2] * fa[2]);
    psi[1] += scale * fs[3] * (R[3] * fa[0] + R[4] * fa[1] + R[5] * fa[2]);
    psi[2] += scale * fs[3] * (R[6] * fa[0] + R[7] * fa[1] + R[8] * fa[2]);
  }
}

/*
  Based on the displacement vector
*/
void BeamTransfer::transferDisps(const F2FScalar *struct_disps,
                                 F2FScalar *aero_disps) {
  // Check if struct nodes locations need to be redistributed
  distributeStructuralMesh();

  // Copy prescribed displacements into displacement vector
  structGatherBcast(dof_per_node * ns_local, struct_disps, dof_per_node * ns,
                    Us);

  for (int i = 0; i < na; i++) {
    // Get the element and parametric location within the element
    // where the aerodynamic node is attached.
    int elem = aero_pt_to_elem[i];
    double xi = aero_pt_to_param[i];

    // Evaluate the shape functions. Save the number of nodes in each
    // element.
    double N[3];
    if (order == 2) {
      N[0] = 0.5 * (1.0 - xi);
      N[1] = 0.5 * (1.0 + xi);
    } else if (order == 3) {
      N[0] = 0.5 * xi * (xi - 1.0);
      N[1] = 1.0 - xi * xi;
      N[2] = 0.5 * xi * (xi + 1.0);
    }

    // Evaluate the average of the structural displacements
    const int *nodes = &conn[order * elem];

    // Zero the aerodynamic displacements
    F2FScalar *ua = &aero_disps[3 * i];
    ua[0] = ua[1] = ua[2] = 0.0;

    // Compute the displacement
    for (int k = 0; k < order; k++, nodes++) {
      const F2FScalar *u = &Us[dof_per_node * nodes[0]];
      ua[0] += N[k] * u[0];
      ua[1] += N[k] * u[1];
      ua[2] += N[k] * u[2];

      // Compute the position vector from the aerodynamic point to the
      // node location
      F2FScalar d[3];
      d[0] = Xa[3 * i] - Xs[3 * nodes[0]];
      d[1] = Xa[3 * i + 1] - Xs[3 * nodes[0] + 1];
      d[2] = Xa[3 * i + 2] - Xs[3 * nodes[0] + 2];

      // Compute the contribution from the rigid transformation
      F2FScalar dr[3];
      computeRotation(&u[3], d, dr);
      ua[0] += N[k] * dr[0];
      ua[1] += N[k] * dr[1];
      ua[2] += N[k] * dr[2];
    }
  }
}

/*
  Transfer the aerodynamic loads to the structural node locations
*/
void BeamTransfer::transferLoads(const F2FScalar *aero_loads,
                                 F2FScalar *struct_loads) {
  // Copy prescribed aero loads into member variable
  memcpy(Fa, aero_loads, 3 * na * sizeof(F2FScalar));

  // Zero struct loads
  F2FScalar *struct_loads_global = new F2FScalar[dof_per_node * ns];
  memset(struct_loads_global, 0, dof_per_node * ns * sizeof(F2FScalar));

  for (int i = 0; i < na; i++) {
    // Get the element and parametric location within the element
    // where the aerodynamic node is attached.
    int elem = aero_pt_to_elem[i];
    double xi = aero_pt_to_param[i];

    // Evaluate the shape functions. Save the number of nodes in each
    // element.
    double N[3];
    if (order == 2) {
      N[0] = 0.5 * (1.0 - xi);
      N[1] = 0.5 * (1.0 + xi);
    } else if (order == 3) {
      N[0] = 0.5 * xi * (xi - 1.0);
      N[1] = 1.0 - xi * xi;
      N[2] = 0.5 * xi * (xi + 1.0);
    }

    // Evaluate the average of the structural displacements
    const int *nodes = &conn[order * elem];

    // Get the aerodynamic loads
    const F2FScalar *fa = &Fa[3 * i];

    // Compute the displacements at each node
    for (int k = 0; k < order; k++, nodes++) {
      const F2FScalar *u = &Us[dof_per_node * nodes[0]];
      F2FScalar *fs = &struct_loads_global[dof_per_node * nodes[0]];
      fs[0] += N[k] * fa[0];
      fs[1] += N[k] * fa[1];
      fs[2] += N[k] * fa[2];

      // Compute the position vector from the aerodynamic point to the
      // node location
      F2FScalar d[3];
      d[0] = Xa[3 * i] - Xs[3 * nodes[0]];
      d[1] = Xa[3 * i + 1] - Xs[3 * nodes[0] + 1];
      d[2] = Xa[3 * i + 2] - Xs[3 * nodes[0] + 2];

      // Compute the contribution from the rigid transformation
      addTransposeRotationDeriv(N[k], &u[3], d, fa, &fs[3]);
    }
  }

  // distribute the structural loads
  structAddScatter(dof_per_node * ns, struct_loads_global,
                   dof_per_node * ns_local, struct_loads);

  delete[] struct_loads_global;
}

// Action of transpose Jacobians needed for solving adjoint system
void BeamTransfer::applydDduS(const F2FScalar *vecs, F2FScalar *prods) {
  // Make a global image of the input vector
  F2FScalar *vecs_global = new F2FScalar[dof_per_node * ns];
  structGatherBcast(dof_per_node * ns_local, vecs, dof_per_node * ns,
                    vecs_global);

  // Zero array of Jacobian-vector products every call
  memset(prods, 0, 3 * na * sizeof(F2FScalar));

  for (int i = 0; i < na; i++) {
    // Get the element and parametric location within the element
    // where the aerodynamic node is attached.
    int elem = aero_pt_to_elem[i];
    double xi = aero_pt_to_param[i];

    // Evaluate the shape functions. Save the number of nodes in each
    // element.
    double N[3];
    if (order == 2) {
      N[0] = 0.5 * (1.0 - xi);
      N[1] = 0.5 * (1.0 + xi);
    } else if (order == 3) {
      N[0] = 0.5 * xi * (xi - 1.0);
      N[1] = 1.0 - xi * xi;
      N[2] = 0.5 * xi * (xi + 1.0);
    }

    // Evaluate the average of the structural displacements
    const int *nodes = &conn[order * elem];

    // Zero the aerodynamic displacements
    F2FScalar *ua = &prods[3 * i];
    ua[0] = ua[1] = ua[2] = 0.0;

    // Compute the displacement
    for (int k = 0; k < order; k++, nodes++) {
      const F2FScalar *u = &Us[dof_per_node * nodes[0]];
      const F2FScalar *v = &vecs_global[dof_per_node * nodes[0]];
      ua[0] -= N[k] * v[0];
      ua[1] -= N[k] * v[1];
      ua[2] -= N[k] * v[2];

      // Compute the position vector from the aerodynamic point to the
      // node location
      F2FScalar d[3];
      d[0] = Xa[3 * i] - Xs[3 * nodes[0]];
      d[1] = Xa[3 * i + 1] - Xs[3 * nodes[0] + 1];
      d[2] = Xa[3 * i + 2] - Xs[3 * nodes[0] + 2];

      // Compute the contribution from the rigid transformation
      F2FScalar dr[3];
      computeRotationDerivProduct(&v[3], &u[3], d, dr);
      ua[0] -= N[k] * dr[0];
      ua[1] -= N[k] * dr[1];
      ua[2] -= N[k] * dr[2];
    }
  }

  // Clean up the allocated memory
  delete[] vecs_global;
}

/*
  Compute the transpose Jacobian-vector product of the displacement transfer
  to the aerodynamic nodes with respect to the structural displacement
  vector.

  This computes the transpose operation, so the input is the size of the
  aerodynamic node vector while the output is the size of the structural node
  locations
*/
void BeamTransfer::applydDduSTrans(const F2FScalar *vecs, F2FScalar *prods) {
  // Zero array of transpose Jacobian-vector products every call
  F2FScalar *prods_global = new F2FScalar[dof_per_node * ns];
  memset(prods_global, 0, dof_per_node * ns * sizeof(F2FScalar));

  for (int i = 0; i < na; i++) {
    // Get the element and parametric location within the element
    // where the aerodynamic node is attached.
    int elem = aero_pt_to_elem[i];
    double xi = aero_pt_to_param[i];

    // Evaluate the shape functions. Save the number of nodes in each
    // element.
    double N[3];
    if (order == 2) {
      N[0] = 0.5 * (1.0 - xi);
      N[1] = 0.5 * (1.0 + xi);
    } else if (order == 3) {
      N[0] = 0.5 * xi * (xi - 1.0);
      N[1] = 1.0 - xi * xi;
      N[2] = 0.5 * xi * (xi + 1.0);
    }

    // Evaluate the average of the structural displacements
    const int *nodes = &conn[order * elem];

    // Zero the aerodynamic displacements
    const F2FScalar *ua = &vecs[3 * i];

    // Compute the displacement
    for (int k = 0; k < order; k++, nodes++) {
      const F2FScalar *u = &Us[dof_per_node * nodes[0]];

      // Get the output
      F2FScalar *p = &prods_global[dof_per_node * nodes[0]];
      p[0] -= N[k] * ua[0];
      p[1] -= N[k] * ua[1];
      p[2] -= N[k] * ua[2];

      // Compute the position vector from the aerodynamic point to the
      // node location
      F2FScalar d[3];
      d[0] = Xa[3 * i] - Xs[3 * nodes[0]];
      d[1] = Xa[3 * i + 1] - Xs[3 * nodes[0] + 1];
      d[2] = Xa[3 * i + 2] - Xs[3 * nodes[0] + 2];

      // Compute the contribution from the rigid transformation
      addTransposeRotationDeriv(-N[k], &u[3], d, ua, &p[3]);
    }
  }
  // distribute the results to the structural processors
  structAddScatter(dof_per_node * ns, prods_global, dof_per_node * ns_local,
                   prods);

  // clean up allocated memory
  delete[] prods_global;
}

void BeamTransfer::applydLduS(const F2FScalar *vecs, F2FScalar *prods) {
  if (dof_per_node == 6) {
    memset(prods, 0, dof_per_node * ns_local * sizeof(F2FScalar));
  } else {
    F2FScalar *vecs_global = new F2FScalar[dof_per_node * ns];
    structGatherBcast(dof_per_node * ns_local, vecs, dof_per_node * ns,
                      vecs_global);

    // Zero products
    F2FScalar *prods_global = new F2FScalar[dof_per_node * ns];
    memset(prods_global, 0, dof_per_node * ns * sizeof(F2FScalar));

    for (int i = 0; i < na; i++) {
      // Get the element and parametric location within the element
      // where the aerodynamic node is attached.
      int elem = aero_pt_to_elem[i];
      double xi = aero_pt_to_param[i];

      // Evaluate the shape functions. Save the number of nodes in each
      // element.
      double N[3];
      if (order == 2) {
        N[0] = 0.5 * (1.0 - xi);
        N[1] = 0.5 * (1.0 + xi);
      } else if (order == 3) {
        N[0] = 0.5 * xi * (xi - 1.0);
        N[1] = 1.0 - xi * xi;
        N[2] = 0.5 * xi * (xi + 1.0);
      }

      // Evaluate the average of the structural displacements
      const int *nodes = &conn[order * elem];

      // Get the aerodynamic loads
      const F2FScalar *fa = &Fa[3 * i];

      // Compute the displacements at each node
      for (int k = 0; k < order; k++, nodes++) {
        const F2FScalar *v = &vecs_global[dof_per_node * nodes[0]];
        F2FScalar *fs = &prods_global[dof_per_node * nodes[0]];

        // Compute the position vector from the aerodynamic point to the
        // node location
        F2FScalar d[3];
        d[0] = Xa[3 * i] - Xs[3 * nodes[0]];
        d[1] = Xa[3 * i + 1] - Xs[3 * nodes[0] + 1];
        d[2] = Xa[3 * i + 2] - Xs[3 * nodes[0] + 2];

        // Compute the contribution from the rigid transformation
        addTransposeRotationDeriv(-N[k], &v[3], d, fa, &fs[3]);
      }
    }

    // distribute the results to the structural processors
    structAddScatter(dof_per_node * ns, prods_global, dof_per_node * ns_local,
                     prods);

    // clean up allocated memory
    delete[] vecs_global;
    delete[] prods_global;
  }
}

void BeamTransfer::applydLduSTrans(const F2FScalar *vecs, F2FScalar *prods) {
  if (dof_per_node == 6) {
    memset(prods, 0, dof_per_node * ns_local * sizeof(F2FScalar));
  } else {
    F2FScalar *vecs_global = new F2FScalar[dof_per_node * ns];
    structGatherBcast(dof_per_node * ns_local, vecs, dof_per_node * ns,
                      vecs_global);

    // Zero products every call
    F2FScalar *prods_global = new F2FScalar[dof_per_node * ns];
    memset(prods_global, 0, dof_per_node * ns * sizeof(F2FScalar));

    for (int i = 0; i < na; i++) {
      // Get the element and parametric location within the element
      // where the aerodynamic node is attached.
      int elem = aero_pt_to_elem[i];
      double xi = aero_pt_to_param[i];

      // Evaluate the shape functions. Save the number of nodes in each
      // element.
      double N[3];
      if (order == 2) {
        N[0] = 0.5 * (1.0 - xi);
        N[1] = 0.5 * (1.0 + xi);
      } else if (order == 3) {
        N[0] = 0.5 * xi * (xi - 1.0);
        N[1] = 1.0 - xi * xi;
        N[2] = 0.5 * xi * (xi + 1.0);
      }

      // Evaluate the average of the structural displacements
      const int *nodes = &conn[order * elem];

      // Get the aerodynamic loads
      const F2FScalar *fa = &Fa[3 * i];

      // Compute the displacements at each node
      for (int k = 0; k < order; k++, nodes++) {
        const F2FScalar *v = &vecs_global[dof_per_node * nodes[0]];
        F2FScalar *fs = &prods_global[dof_per_node * nodes[0]];

        // Compute the position vector from the aerodynamic point to the
        // node location
        F2FScalar d[3];
        d[0] = Xa[3 * i] - Xs[3 * nodes[0]];
        d[1] = Xa[3 * i + 1] - Xs[3 * nodes[0] + 1];
        d[2] = Xa[3 * i + 2] - Xs[3 * nodes[0] + 2];

        // Compute the contribution from the rigid transformation
        addTransposeRotationDeriv(-N[k], &v[3], d, fa, &fs[3]);
      }
    }

    // distribute the results to the structural processors
    structAddScatter(dof_per_node * ns, prods_global, dof_per_node * ns_local,
                     prods);

    // clean up allocated memory
    delete[] vecs_global;
    delete[] prods_global;
  }
}

// Action of Jacobians needed for assembling gradient from adjoint variables
void BeamTransfer::applydDdxA0(const F2FScalar *vecs, F2FScalar *prods) {
  memset(prods, 0, 3 * na * sizeof(F2FScalar));

  for (int i = 0; i < na; i++) {
    // Get the element and parametric location within the element
    // where the aerodynamic node is attached.
    int elem = aero_pt_to_elem[i];
    double xi = aero_pt_to_param[i];

    // Evaluate the shape functions. Save the number of nodes in each
    // element.
    double N[3];
    if (order == 2) {
      N[0] = 0.5 * (1.0 - xi);
      N[1] = 0.5 * (1.0 + xi);
    } else if (order == 3) {
      N[0] = 0.5 * xi * (xi - 1.0);
      N[1] = 1.0 - xi * xi;
      N[2] = 0.5 * xi * (xi + 1.0);
    }

    // Evaluate the average of the structural displacements
    const int *nodes = &conn[order * elem];

    // Zero the aerodynamic displacements
    const F2FScalar *ua = &vecs[3 * i];

    // Compute the displacement
    for (int k = 0; k < order; k++, nodes++) {
      const F2FScalar *u = &Us[dof_per_node * nodes[0]];

      // Compute the contribution from the rigid transformation
      F2FScalar dr[3];
      computeRotationTranspose(&u[3], ua, dr);
      prods[3 * i] -= N[k] * dr[0];
      prods[3 * i + 1] -= N[k] * dr[1];
      prods[3 * i + 2] -= N[k] * dr[2];
    }
  }
}

void BeamTransfer::applydDdxS0(const F2FScalar *vecs, F2FScalar *prods) {
  // Set products to zero - 3 coordinates per node
  F2FScalar *prods_global = new F2FScalar[3 * ns];
  memset(prods_global, 0.0, 3 * ns * sizeof(F2FScalar));

  for (int i = 0; i < na; i++) {
    // Get the element and parametric location within the element
    // where the aerodynamic node is attached.
    int elem = aero_pt_to_elem[i];
    double xi = aero_pt_to_param[i];

    // Evaluate the shape functions. Save the number of nodes in each
    // element.
    double N[3];
    if (order == 2) {
      N[0] = 0.5 * (1.0 - xi);
      N[1] = 0.5 * (1.0 + xi);
    } else if (order == 3) {
      N[0] = 0.5 * xi * (xi - 1.0);
      N[1] = 1.0 - xi * xi;
      N[2] = 0.5 * xi * (xi + 1.0);
    }

    // Evaluate the average of the structural displacements
    const int *nodes = &conn[order * elem];

    // Zero the aerodynamic displacements
    const F2FScalar *ua = &vecs[3 * i];

    // Compute the displacement
    for (int k = 0; k < order; k++, nodes++) {
      const F2FScalar *u = &Us[dof_per_node * nodes[0]];

      // Compute the contribution from the rigid transformation
      F2FScalar dr[3];
      computeRotationTranspose(&u[3], ua, dr);
      prods_global[3 * nodes[0]] += N[k] * dr[0];
      prods_global[3 * nodes[0] + 1] += N[k] * dr[1];
      prods_global[3 * nodes[0] + 2] += N[k] * dr[2];
    }
  }
  // distribute the results to the structural processors. Here we use
  // 3 node coordinates per node
  structAddScatter(3 * ns, prods_global, 3 * ns_local, prods);

  // clean up allocated memory
  delete[] prods_global;
}

void BeamTransfer::applydLdxA0(const F2FScalar *vecs, F2FScalar *prods) {
  F2FScalar *vecs_global = new F2FScalar[dof_per_node * ns];
  structGatherBcast(dof_per_node * ns_local, vecs, dof_per_node * ns,
                    vecs_global);

  // Zero products
  memset(prods, 0, 3 * na * sizeof(F2FScalar));
  for (int i = 0; i < na; i++) {
    // Get the element and parametric location within the element
    // where the aerodynamic node is attached.
    int elem = aero_pt_to_elem[i];
    double xi = aero_pt_to_param[i];

    // Evaluate the shape functions. Save the number of nodes in each
    // element.
    double N[3];
    if (order == 2) {
      N[0] = 0.5 * (1.0 - xi);
      N[1] = 0.5 * (1.0 + xi);
    } else if (order == 3) {
      N[0] = 0.5 * xi * (xi - 1.0);
      N[1] = 1.0 - xi * xi;
      N[2] = 0.5 * xi * (xi + 1.0);
    }

    // Evaluate the average of the structural displacements
    const int *nodes = &conn[order * elem];

    // Get the aerodynamic loads
    const F2FScalar *fa = &Fa[3 * i];

    // Compute the displacements at each node
    for (int k = 0; k < order; k++, nodes++) {
      const F2FScalar *u = &Us[dof_per_node * nodes[0]];
      const F2FScalar *fs = &vecs_global[dof_per_node * nodes[0]];

      // Compute the contribution from the rigid transformation
      addTransposeRotationDerivAdjoint(-N[k], &u[3], fa, &fs[3], &prods[3 * i]);
    }
  }
}

void BeamTransfer::applydLdxS0(const F2FScalar *vecs, F2FScalar *prods) {
  F2FScalar *vecs_global = new F2FScalar[dof_per_node * ns];
  structGatherBcast(dof_per_node * ns_local, vecs, dof_per_node * ns,
                    vecs_global);

  // Zero products
  F2FScalar *prods_global = new F2FScalar[3 * ns];
  memset(prods_global, 0, 3 * ns * sizeof(F2FScalar));

  for (int i = 0; i < na; i++) {
    // Get the element and parametric location within the element
    // where the aerodynamic node is attached.
    int elem = aero_pt_to_elem[i];
    double xi = aero_pt_to_param[i];

    // Evaluate the shape functions. Save the number of nodes in each
    // element.
    double N[3];
    if (order == 2) {
      N[0] = 0.5 * (1.0 - xi);
      N[1] = 0.5 * (1.0 + xi);
    } else if (order == 3) {
      N[0] = 0.5 * xi * (xi - 1.0);
      N[1] = 1.0 - xi * xi;
      N[2] = 0.5 * xi * (xi + 1.0);
    }

    // Evaluate the average of the structural displacements
    const int *nodes = &conn[order * elem];

    // Get the aerodynamic loads
    const F2FScalar *fa = &Fa[3 * i];

    // Compute the displacements at each node
    for (int k = 0; k < order; k++, nodes++) {
      const F2FScalar *u = &Us[dof_per_node * nodes[0]];
      const F2FScalar *fs = &vecs_global[dof_per_node * nodes[0]];

      // Compute the contribution from the rigid transformation
      addTransposeRotationDerivAdjoint(N[k], &u[3], fa, &fs[3],
                                       &prods_global[3 * nodes[0]]);
    }
  }

  // distribute the results to the structural processors
  structAddScatter(3 * ns, prods_global, 3 * ns_local, prods);

  // Free allocated memory
  delete[] vecs_global;
  delete[] prods_global;
}
