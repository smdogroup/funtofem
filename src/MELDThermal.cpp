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

#include "MELDThermal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstring>

#include "LocatePoint.h"
#include "funtofemlapack.h"

MELDThermal::MELDThermal(MPI_Comm all, MPI_Comm structure, int _struct_root,
                         MPI_Comm aero, int _aero_root, int symmetry,
                         int num_nearest, F2FScalar beta) {
  // Initialize communicators
  global_comm = all;
  struct_comm = structure;
  aero_comm = aero;
  struct_root = _struct_root;
  aero_root = _aero_root;

  // Initialize aerodynamic data member variables
  Xa = NULL;
  Fa =
      NULL;  // Aerodynamic area-weighted normal thermal flux component (scalar)
  na = 0;

  // Initialize structural data member variables
  Xs = NULL;
  Us = NULL;  // Temperature, not displacements
  ns = 0;

  Xs_local = NULL;
  ns_local = 0;

  // Set the symmetry parameter
  isymm = symmetry;

  // Set the number of structural nodes linked to each aerodynamic node
  nn = num_nearest;

  // Set the global weighting decay parameter
  global_beta = beta;

  // Initialize the aerostuctural connectivity
  global_conn = NULL;
  global_W = NULL;

  // Initialize object id
  object_id = TransferScheme::object_count++;

  // Inintialize the indicator of a mesh update
  mesh_update = 0;

  // Notify user of the type of transfer scheme they are using
  int rank;
  MPI_Comm_rank(global_comm, &rank);
  if (rank == struct_root) {
    printf("Transfer scheme [%i]: Creating scheme of type MELDThermal...\n",
           object_id);
  }
}

MELDThermal::~MELDThermal() {
  // Free the aerostructural connectivity data
  if (global_conn) {
    delete[] global_conn;
  }

  // Free the load transfer data
  if (global_W) {
    delete[] global_W;
  }

  int rank;
  MPI_Comm_rank(global_comm, &rank);
  if (rank == struct_root) {
    printf("Transfer scheme [%i]: freeing MELD data...\n", object_id);
  }
}

void MELDThermal::setStructNodes(const F2FScalar *struct_X, int struct_nnodes) {
  // Free the structural data if any is allocated
  if (Xs_local) {
    delete[] Xs_local;
    Xs_local = NULL;
  }

  ns_local = struct_nnodes;
  Xs_local = new F2FScalar[3 * ns_local];
  memcpy(Xs_local, struct_X, 3 * ns_local * sizeof(F2FScalar));

  mesh_update = 1;
}

/*
  Set the aerodynamic surface node locations
*/
void MELDThermal::setAeroNodes(const F2FScalar *aero_X, int aero_nnodes) {
  na = aero_nnodes;

  // Free the aerodynamic data if any is allocated
  if (Xa) {
    delete[] Xa;
  }
  if (Fa) {
    delete[] Fa;
  }

  // Global number of aerodynamic nodes
  na_global = 0;
  MPI_Allreduce(&na, &na_global, 1, MPI_INT, MPI_SUM, global_comm);

  // Allocate memory for aerodynamic data, copy in node locations, initialize
  // displacement and load arrays
  if (na > 0) {
    Xa = new F2FScalar[3 * na];
    memcpy(Xa, aero_X, 3 * na * sizeof(F2FScalar));

    Fa = new F2FScalar[na];
    memset(Fa, 0, 1 * na * sizeof(F2FScalar));
  }
}

/*
  Collect a structural vector to create a global image then distribute to the
  aerodynamic processors
*/
void MELDThermal::collectStructuralVector(const F2FScalar *local,
                                          F2FScalar *global,
                                          int vars_per_node) {
  // Collect how many structural nodes every processor has
  if (struct_comm != MPI_COMM_NULL) {
    int struct_nprocs;
    int struct_rank;
    MPI_Comm_size(struct_comm, &struct_nprocs);
    MPI_Comm_rank(struct_comm, &struct_rank);

    int *ns_list = new int[struct_nprocs];
    memset(ns_list, 0, struct_nprocs * sizeof(int));

    MPI_Gather(&ns_local, 1, MPI_INT, ns_list, 1, MPI_INT, struct_root,
               struct_comm);

    // Collect the structural nodes on the master
    int send_size = ns_local * vars_per_node;
    int *disps = new int[struct_nprocs];
    memset(disps, 0, struct_nprocs * sizeof(int));

    if (struct_rank == struct_root) {
      for (int proc = 0; proc < struct_nprocs; proc++) {
        ns_list[proc] *= vars_per_node;
        if (proc > 0) {
          disps[proc] = disps[proc - 1] + ns_list[proc - 1];
        }
      }
    }

    MPI_Gatherv(local, send_size, F2F_MPI_TYPE, global, ns_list, disps,
                F2F_MPI_TYPE, 0, struct_comm);

    delete[] ns_list;
    delete[] disps;
  }

  // Pass the global list to all the processors
  MPI_Bcast(global, vars_per_node * ns, F2F_MPI_TYPE, struct_root, global_comm);
}

/*
  Reduce vector to get the total across all aero procs then distribute to the
  structural processors
*/
void MELDThermal::distributeStructuralVector(F2FScalar *global,
                                             F2FScalar *local,
                                             int vars_per_node) {
  // Get the contributions from each aero processor
  MPI_Allreduce(MPI_IN_PLACE, global, ns * vars_per_node, F2F_MPI_TYPE, MPI_SUM,
                global_comm);

  // Collect how many nodes each structural processor has
  if (struct_comm != MPI_COMM_NULL) {
    int struct_nprocs;
    int struct_rank;
    MPI_Comm_size(struct_comm, &struct_nprocs);
    MPI_Comm_rank(struct_comm, &struct_rank);

    int *ns_list = new int[struct_nprocs];

    MPI_Gather(&ns_local, 1, MPI_INT, ns_list, 1, MPI_INT, 0, struct_comm);

    // Distribute to the structural processors
    int *disps = new int[struct_nprocs];
    memset(disps, 0, struct_nprocs * sizeof(int));

    if (struct_rank == 0) {
      for (int proc = 0; proc < struct_nprocs; proc++) {
        ns_list[proc] *= vars_per_node;
        if (proc > 0) {
          disps[proc] = disps[proc - 1] + ns_list[proc - 1];
        }
      }
    }

    MPI_Scatterv(global, ns_list, disps, F2F_MPI_TYPE, local,
                 ns_local * vars_per_node, F2F_MPI_TYPE, 0, struct_comm);

    delete[] ns_list;
    delete[] disps;
  }
}

void MELDThermal::distributeStructuralMesh() {
  MPI_Allreduce(MPI_IN_PLACE, &mesh_update, 1, MPI_INT, MPI_SUM, global_comm);
  if (mesh_update > 0) {
    ns = 0;
    if (struct_comm != MPI_COMM_NULL) {
      MPI_Reduce(&ns_local, &ns, 1, MPI_INT, MPI_SUM, 0, struct_comm);
    }

    MPI_Bcast(&ns, 1, MPI_INT, struct_root, global_comm);

    // Allocate memory for structural data, initialize displacement array
    if (Xs) {
      delete[] Xs;
    }
    if (Us) {
      delete[] Us;
    }

    Xs = new F2FScalar[3 * ns];
    memset(Xs, 0, 3 * ns * sizeof(F2FScalar));

    Us = new F2FScalar[ns];
    memset(Us, 0, ns * sizeof(F2FScalar));

    collectStructuralVector(Xs_local, Xs);
    if (Xs_local) {
      delete[] Xs_local;
      Xs_local = NULL;
    }
    mesh_update = 0;
  }
}

/*
  Set aerostructural connectivity, compute weights, and allocate memory needed
  for transfers and products
*/
void MELDThermal::initialize() {
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
}

/*
  Builds aerostructural connectivity through LocatePoint search, linking each
  aerodynamic node with a specified number of nearest structural nodes

  Return
  --------
  conn : aerostructural connectivity

*/
void MELDThermal::setAeroStructConn(int *conn) {
  // Copy or duplicate and reflect the unique structural nodes
  F2FScalar *Xs_dup = NULL;
  int num_locate_nodes = 0;

  if (isymm > -1) {
    Xs_dup = new F2FScalar[6 * ns];
    memcpy(Xs_dup, Xs, 3 * ns * sizeof(F2FScalar));

    double tol = 1e-7;
    for (int k = 0; k < ns; k++) {
      if (fabs(F2FRealPart(Xs_dup[3 * k + isymm])) > tol) {
        // Node is not on the plane of symmetry, so copy and...
        memcpy(&Xs_dup[3 * (ns + k)], &Xs_dup[3 * k], 3 * sizeof(F2FScalar));
        // reflect
        Xs_dup[3 * (ns + k) + isymm] *= -1.0;
      } else {
        Xs_dup[3 * (ns + k) + 0] = -(k + 1) * 9.0e9;
        Xs_dup[3 * (ns + k) + 1] = -(k + 1) * 9.0e9 + 1;
        Xs_dup[3 * (ns + k) + 2] = -(k + 1) * 9.0e9 + 2;
      }
    }
    num_locate_nodes = 2 * ns;
  } else {
    Xs_dup = new F2FScalar[3 * ns];
    memcpy(Xs_dup, Xs, 3 * ns * sizeof(F2FScalar));
    num_locate_nodes = ns;
  }

  // Create instance of LocatePoint class to perform the following searches
  int min_bin_size = 10;
  // bug is in one of these variables:
  LocatePoint *locator =
      new LocatePoint(Xs_dup, num_locate_nodes, min_bin_size);

  // Indices of nearest nodes
  int *indx = new int[nn];
  F2FScalar *dist = new F2FScalar[nn];

  // For each aerodynamic node, copy the indices of the nearest n structural
  // nodes into the conn array
  for (int i = 0; i < na; i++) {
    F2FScalar xa0[3];
    memcpy(xa0, &Xa[3 * i], 3 * sizeof(F2FScalar));

    locator->locateKClosest(nn, indx, dist, xa0);
    memcpy(&conn[nn * i], indx, nn * sizeof(int));
  }

  // Free the duplicate array
  delete[] Xs_dup;

  // Delete the LocatePoint object and release memory
  delete[] indx;
  delete[] dist;
  delete locator;
}

/*
  Computes weights of structural nodes

  Returns
  --------
  W : weights

*/
void MELDThermal::computeWeights(F2FScalar *W) {
  for (int i = 0; i < na; i++) {
    const F2FScalar *xa0 = &Xa[3 * i];
    const int *local_conn = &global_conn[i * nn];
    F2FScalar *w = &W[i * nn];

    // Compute the weights based on the difference between the aerodynamic point
    // location and the local undeformed structural point
    F2FScalar wtotal = 0.0;
    F2FScalar v[3];

    // Find the average squared distance for normalization
    F2FScalar v2_avg = 0.0;
    F2FScalar dist;

    for (int j = 0; j < nn; j++) {
      if (local_conn[j] < ns) {
        const F2FScalar *xs0 = &Xs[3 * local_conn[j]];
        vec_diff(xs0, xa0, v);
      } else {
        F2FScalar rxs0[3];  // Reflected xs0
        memcpy(rxs0, &Xs[3 * (local_conn[j] - ns)], 3 * sizeof(F2FScalar));
        rxs0[isymm] *= -1.0;
        vec_diff(rxs0, xa0, v);
      }
      v2_avg += vec_dot(v, v) / (1.0 * nn);
    }

    // Make sure we don't divide by zero
    if (F2FRealPart(v2_avg) < 1.0e-7) {
      v2_avg = 1.0e-7;
    }

    for (int j = 0; j < nn; j++) {
      if (local_conn[j] < ns) {
        const F2FScalar *xs0 = &Xs[3 * local_conn[j]];
        vec_diff(xs0, xa0, v);
      } else {
        F2FScalar rxs0[3];  // Reflected xs0
        memcpy(rxs0, &Xs[3 * (local_conn[j] - ns)], 3 * sizeof(F2FScalar));
        rxs0[isymm] *= -1.0;
        vec_diff(rxs0, xa0, v);
      }
      w[j] = exp(-global_beta * vec_dot(v, v) / v2_avg);
      wtotal += w[j];
    }

    // Normalize the weights
    wtotal = 1.0 / wtotal;
    for (int j = 0; j < nn; j++) {
      w[j] *= wtotal;
    }
  }
}

/*
  Computes the displacements of aerodynamic surface nodes by fitting an
  optimal rigid rotation and translation to the displacement of the set of
  structural nodes nearest each aerodynamic surface node

  Arguments
  ---------
  struct_disps : structural node displacements

  Returns
  -------
  aero_disps   : aerodynamic node displacements

*/
void MELDThermal::transferTemp(const F2FScalar *struct_Temp,
                               F2FScalar *aero_Temp) {
  // Distribute the mesh components if needed
  distributeStructuralMesh();

  // Copy the temperature into the global temperature vector
  collectStructuralVector(struct_Temp, Us,
                          1);  // set vars_per_node = 1 for temps

  // Zero the outputs
  memset(aero_Temp, 0.0, na * sizeof(F2FScalar));

  for (int i = 0; i < na; i++) {
    const int *local_conn = &global_conn[i * nn];
    const F2FScalar *w = &global_W[i * nn];

    F2FScalar Taero = 0.0;
    for (int j = 0; j < nn; j++) {
      if (local_conn[j] < ns) {
        Taero += w[j] * Us[local_conn[j]];
      } else {
        Taero += w[j] * Us[local_conn[j] - ns];
      }
    }

    aero_Temp[i] = Taero;
  }
}

/*
  Computes the loads on all structural nodes consistently and conservatively
  from loads on aerodynamic surface nodes

  Arguments
  ---------
  aero_flux   : normal flux through surface on aerodynamic surface nodes

  Returns
  -------
  struct_loads : loads on structural nodes

*/
void MELDThermal::transferFlux(const F2FScalar *aero_flux,
                               F2FScalar *struct_flux) {
  // Copy prescribed aero loads into member variable
  memcpy(Fa, aero_flux, na * sizeof(F2FScalar));

  // Zero struct flux
  F2FScalar *struct_flux_global = new F2FScalar[ns];
  memset(struct_flux_global, 0, ns * sizeof(F2FScalar));

  for (int i = 0; i < na; i++) {
    const int *local_conn = &global_conn[i * nn];
    const F2FScalar *w = &global_W[i * nn];
    const F2FScalar *fa = &Fa[i];

    for (int j = 0; j < nn; j++) {
      int index = 0;
      if (local_conn[j] < ns) {
        index = local_conn[j];
      } else {
        index = local_conn[j] - ns;
      }
      struct_flux_global[index] += w[j] * fa[0];
    }
  }

  // set vars_per_node = 1 for flux
  distributeStructuralVector(struct_flux_global, struct_flux, 1);
  delete[] struct_flux_global;
}

/*
  Apply the action of the temperature transfer w.r.t structural temperature
  Jacobian to the input vector

  Arguments
  ---------
  vecs  : input vector

  Returns
  --------
  prods : output vector

*/
void MELDThermal::applydTdtS(const F2FScalar *vecs, F2FScalar *prods) {
  // Make a global image of the input vector
  F2FScalar *vecs_global = new F2FScalar[ns];
  collectStructuralVector(vecs, vecs_global,
                          1);  // set vars_per_node = 1 for temps

  // Zero array of Jacobian-vector products every call
  memset(prods, 0, na * sizeof(F2FScalar));

  // Loop over all aerodynamic surface nodes
  for (int i = 0; i < na; i++) {
    // Loop over linked structural nodes and add up nonzero contributions to
    // Jacobian-vector product
    for (int j = 0; j < nn; j++) {
      int indx = global_conn[nn * i + j];

      F2FScalar v;
      if (indx < ns) {
        v = vecs_global[indx];
      } else {
        indx -= ns;
        v = vecs_global[indx];
      }

      // Compute each component of the Jacobian vector product as follows:
      // Jv[k] = w*v[k]
      F2FScalar w = global_W[nn * i + j];
      prods[i] -= w * v;
    }
  }

  // Clean up the allocated memory
  delete[] vecs_global;
}

/*
  Apply the action of the temperature transfer w.r.t structural temperature
  transpose Jacobian to the input vector

  Arguments
  ----------
  vecs  : input vector

  Returns
  --------
  prods : output vector

*/
void MELDThermal::applydTdtSTrans(const F2FScalar *vecs, F2FScalar *prods) {
  // Zero array of transpose Jacobian-vector products every call
  F2FScalar *prods_global = new F2FScalar[ns];
  memset(prods_global, 0, ns * sizeof(F2FScalar));

  // Loop over aerodynamic surface nodes
  for (int i = 0; i < na; i++) {
    // Loop over linked structural nodes and add up nonzero contributions to
    // Jacobian-vector product
    for (int j = 0; j < nn; j++) {
      int indx = global_conn[nn * i + j];
      F2FScalar w = global_W[nn * i + j];

      if (indx < ns) {
        prods_global[indx] -= w * vecs[i];
      } else {
        indx -= ns;
        prods_global[indx] -= w * vecs[i];
      }
    }
  }

  // distribute the results to the structural processors
  distributeStructuralVector(prods_global, prods,
                             1);  // set vars_per_node = 1 for flux

  // clean up allocated memory
  delete[] prods_global;
}

/*
  Apply the action of the flux transfer w.r.t structural temperature
  Jacobian to the input vector

  Arguments
  ---------
  vecs  : input vector

  Returns
  --------
  prods : output vector

*/
void MELDThermal::applydQdqA(const F2FScalar *vecs, F2FScalar *prods) {
  applydTdtSTrans(vecs, prods);
}

/*
  Apply the action of the flux transfer w.r.t structural temperature
  transpose Jacobian to the input vector

  Arguments
  ----------
  vecs  : input vector

  Returns
  --------
  prods : output vector

*/
void MELDThermal::applydQdqATrans(const F2FScalar *vecs, F2FScalar *prods) {
  applydTdtS(vecs, prods);
}

/*
  Tests flux transfer by computing derivative of product of heat flux on and
  temperatures of aerodynamic surface nodes with respect to structural node
  temperatures and comparing with results from finite difference and complex
  step approximation

  Arguments
  ---------
  struct_temps : structural node temperatures
  aero_flux    : heat flux on aerodynamic surface nodes
  pert         : direction of perturbation of structural node heat flux
  h            : step size

*/
void MELDThermal::testFluxTransfer(const F2FScalar *struct_temps,
                                   const F2FScalar *aero_flux,
                                   const F2FScalar *pert, const F2FScalar h) {
  // Transfer the structural temperatures
  F2FScalar *aero_temps = new F2FScalar[na];
  transferTemp(struct_temps, aero_temps);

  // Transfer the aerodynamic heat flux
  F2FScalar *struct_flux = new F2FScalar[ns];
  transferFlux(aero_flux, struct_flux);

  // Compute directional derivative (structural heat flux times perturbation)
  F2FScalar deriv = 0.0;
  for (int j = 0; j < ns; j++) {
    deriv += struct_flux[j] * pert[j];
  }

  // Approximate using complex step
#ifdef FUNTOFEM_USE_COMPLEX
  F2FScalar *Us_cs = new F2FScalar[ns];
  F2FScalar *Ua_cs = new F2FScalar[na];

  for (int j = 0; j < ns; j++) {
    Us_cs[j] =
        struct_temps[j] + F2FScalar(0.0, F2FRealPart(h) * F2FRealPart(pert[j]));
  }
  transferTemp(Us_cs, Ua_cs);

  F2FScalar work = 0.0;
  for (int i = 0; i < na; i++) {
    work += Fa[i] * Ua_cs[i];
  }
  F2FScalar deriv_approx = F2FImagPart(work) / h;

  delete[] Us_cs;
  delete[] Ua_cs;

  // Approximate using finite difference (central)
#else
  F2FScalar *Us_pos = new F2FScalar[ns];
  F2FScalar *Us_neg = new F2FScalar[ns];
  F2FScalar *Ua_pos = new F2FScalar[na];
  F2FScalar *Ua_neg = new F2FScalar[na];
  for (int j = 0; j < ns; j++) {
    Us_pos[j] = struct_temps[j] + h * pert[j];
    Us_neg[j] = struct_temps[j] - h * pert[j];
  }

  transferTemp(Us_pos, Ua_pos);
  F2FScalar work_pos = 0.0;
  for (int i = 0; i < na; i++) {
    work_pos += Fa[i] * Ua_pos[i];
  }

  transferTemp(Us_neg, Ua_neg);
  F2FScalar work_neg = 0.0;
  for (int i = 0; i < na; i++) {
    work_neg += Fa[i] * Ua_neg[i];
  }

  F2FScalar deriv_approx = 0.5 * (work_pos - work_neg) / h;

  delete[] Us_pos;
  delete[] Us_neg;
  delete[] Ua_pos;
  delete[] Ua_neg;
#endif  // FUNTOFEM_USE_COMPLEX
  // Compute relative error
  F2FScalar rel_error = (deriv - deriv_approx) / deriv_approx;

  // Print results
  printf("\n");
  printf("Heat Flux transfer test with step: %e\n", F2FRealPart(h));
  printf("deriv          = %22.15e\n", F2FRealPart(deriv));
  printf("deriv, approx  = %22.15e\n", F2FRealPart(deriv_approx));
  printf("relative error = %22.15e\n", F2FRealPart(rel_error));
  printf("\n");

  // Free allocated memory
  delete[] aero_temps;
  delete[] struct_flux;
}

/*
  Tests output of dTdtSProducts and dTdtSTransProducts by computing a product
  test_vec_a*J*test_vec_s (where J is the Jacobian) and comparing with results
  from finite difference and complex step

  Arguments
  ---------
  struct_temps : structural node temperatures
  test_vec_a   : test vector the length of the aero nodes
  test_vec_s   : test vector the length of the struct nodes
  h            : step size

*/
void MELDThermal::testTempJacVecProducts(const F2FScalar *struct_temps,
                                         const F2FScalar *test_vec_a,
                                         const F2FScalar *test_vec_s,
                                         const F2FScalar h) {
  // Transfer the structural temperatures
  F2FScalar *aero_temps = new F2FScalar[na];
  transferTemp(struct_temps, aero_temps);

  // Compute the Jacobian-vector products using the function
  F2FScalar *grad1 = new F2FScalar[na];
  applydTdtS(test_vec_s, grad1);

  // Compute product of test_vec_a with the Jacobian-vector products
  F2FScalar deriv1 = 0.0;
  for (int i = 0; i < na; i++) {
    deriv1 += test_vec_a[i] * grad1[i];
  }

  // Compute the transpose Jacobian-vector products using the function
  F2FScalar *grad2 = new F2FScalar[ns];
  applydTdtSTrans(test_vec_a, grad2);

  // Compute product of V1 with the transpose Jacobian-vector products
  F2FScalar deriv2 = 0.0;
  for (int j = 0; j < ns; j++) {
    deriv2 += test_vec_s[j] * grad2[j];
  }

  // Compute complex step approximation
#ifdef FUNTOFEM_USE_COMPLEX
  F2FScalar *Us_cs = new F2FScalar[ns];
  memset(Us_cs, 0.0, ns * sizeof(F2FScalar));
  for (int j = 0; j < ns; j++) {
    Us_cs[j] +=
        struct_temps[j] + F2FScalar(0.0, F2FRealPart(h * test_vec_s[j]));
  }
  transferTemp(Us_cs, aero_temps);

  F2FScalar VPsi = 0.0;
  for (int i = 0; i < na; i++) {
    F2FScalar Psi = Xa[i] + aero_temps[i];
    VPsi += test_vec_a[i] * Psi;
  }
  F2FScalar deriv1_approx = -1.0 * F2FImagPart(VPsi) / h;

  delete[] Us_cs;

  // Compute finite difference approximation (central)
#else
  F2FScalar *Us_pos = new F2FScalar[ns];
  F2FScalar *Us_neg = new F2FScalar[ns];
  for (int j = 0; j < ns; j++) {
    Us_pos[j] = struct_temps[j] + h * test_vec_s[j];
    Us_neg[j] = struct_temps[j] - h * test_vec_s[j];
  }

  transferTemp(Us_pos, aero_temps);
  F2FScalar VPsi_pos = 0.0;
  for (int i = 0; i < na; i++) {
    F2FScalar Psi = Xa[i] + aero_temps[i];
    VPsi_pos += test_vec_a[i] * Psi;
  }

  transferTemp(Us_neg, aero_temps);
  F2FScalar VPsi_neg = 0.0;
  for (int i = 0; i < na; i++) {
    F2FScalar Psi = Xa[i] + aero_temps[i];
    VPsi_neg += test_vec_a[i] * Psi;
  }

  F2FScalar deriv1_approx = -0.5 * (VPsi_pos - VPsi_neg) / h;

  delete[] Us_pos;
  delete[] Us_neg;
#endif
  // Compute relative error
  F2FScalar rel_error1 = (deriv1 - deriv1_approx) / deriv1_approx;
  F2FScalar rel_error2 = (deriv2 - deriv1_approx) / deriv1_approx;

  // Print out results of test
  printf("V2^{T}*dT/dt_{S}*V1 test with step: %e\n", F2FRealPart(h));
  printf("deriv          = %22.15e\n", F2FRealPart(deriv1));
  printf("deriv, approx  = %22.15e\n", F2FRealPart(deriv1_approx));
  printf("relative error = %22.15e\n", F2FRealPart(rel_error1));
  printf("\n");

  printf("V1^{T}*(dT/dt_{S})^{T}*V2 test with step: %e\n", F2FRealPart(h));
  printf("deriv          = %22.15e\n", F2FRealPart(deriv2));
  printf("deriv, approx  = %22.15e\n", F2FRealPart(deriv1_approx));
  printf("relative error = %22.15e\n", F2FRealPart(rel_error2));
  printf("\n");

  // Free allocated memory
  delete[] aero_temps;
  delete[] grad1;
  delete[] grad2;
}

/*
  Tests output of dQdqAProducts and dQdqATransProducts by computing a product
  test_vec_a*J*test_vec_s (where J is the Jacobian) and comparing with
  results from finite difference and complex step

  Arguments
  ---------
  struct_temps : structural node temperatures
  aero_flux   : aerodynamic heat flux
  test_vec_a  : test vector the size of aero nodes
  test_vec_s  : test vector the size of struct nodes
  h            : step size

*/
void MELDThermal::testFluxJacVecProducts(const F2FScalar *struct_temps,
                                         const F2FScalar *aero_flux,
                                         const F2FScalar *test_vec_a,
                                         const F2FScalar *test_vec_s,
                                         const F2FScalar h) {
  // Transfer the structural displacements
  F2FScalar *aero_temps = new F2FScalar[na];
  transferTemp(struct_temps, aero_temps);

  // Transfer the aerodynamic loads to get MM, IPIV
  F2FScalar *struct_flux = new F2FScalar[ns];
  transferFlux(aero_flux, struct_flux);

  // Compute the Jacobian-vector products using the function
  F2FScalar *grad1 = new F2FScalar[ns];
  applydQdqA(test_vec_a, grad1);

  // Compute directional derivative
  F2FScalar deriv1 = 0.0;
  for (int j = 0; j < ns; j++) {
    deriv1 += grad1[j] * test_vec_s[j];
  }

  // Compute transpose Jacobian-vector products using the function
  F2FScalar *grad2 = new F2FScalar[na];
  applydQdqATrans(test_vec_s, grad2);

  // Compute directional derivative
  F2FScalar deriv2 = 0.0;
  for (int j = 0; j < na; j++) {
    deriv2 += grad2[j] * test_vec_a[j];
  }

  // Approximate using complex step
#ifdef FUNTOFEM_USE_COMPLEX
  F2FScalar *Us_cs = new F2FScalar[ns];
  memset(Us_cs, 0.0, ns * sizeof(F2FScalar));
  for (int j = 0; j < ns; j++) {
    Us_cs[j] += struct_temps[j] +
                F2FScalar(0.0, F2FRealPart(h) * F2FRealPart(test_vec_s[j]));
    // F2FScalar(0.0, F2FRealPart(h*test_vec_s1[j]));
  }
  transferTemp(Us_cs, aero_temps);
  transferFlux(aero_flux, struct_flux);

  F2FScalar VPhi = 0.0;
  for (int j = 0; j < na; j++) {
    F2FScalar Phi = Xa[j] + aero_temps[j];
    VPhi += test_vec_a[j] * Phi;
  }
  F2FScalar deriv_approx = -1.0 * F2FImagPart(VPhi) / h;
  delete[] Us_cs;

  // Approximate using finite difference (central)
#else
  F2FScalar *Us_pos = new F2FScalar[ns];
  F2FScalar *Us_neg = new F2FScalar[ns];
  for (int j = 0; j < ns; j++) {
    Us_pos[j] = struct_temps[j] + h * test_vec_a[j];
    Us_neg[j] = struct_temps[j] - h * test_vec_a[j];
  }

  transferTemp(Us_pos, aero_temps);
  transferFlux(aero_flux, struct_flux);
  F2FScalar VPhi_pos = 0.0;
  for (int j = 0; j < ns; j++) {
    F2FScalar Phi = struct_flux[j];
    VPhi_pos += test_vec_s[j] * Phi;
  }

  transferTemp(Us_neg, aero_temps);
  transferFlux(aero_flux, struct_flux);
  F2FScalar VPhi_neg = 0.0;
  for (int j = 0; j < ns; j++) {
    F2FScalar Phi = struct_flux[j];
    VPhi_neg += test_vec_s[j] * Phi;
  }

  F2FScalar deriv_approx = -0.5 * (VPhi_pos - VPhi_neg) / h;

  delete[] Us_pos;
  delete[] Us_neg;
#endif
  // Compute relative error
  F2FScalar rel_error1 = (deriv1 - deriv_approx) / deriv_approx;
  F2FScalar rel_error2 = (deriv2 - deriv_approx) / deriv_approx;

  // Print out results of test
  printf("V2^{T}*dQ/dq_{A}*V1 test with step: %e\n", F2FRealPart(h));
  printf("deriv          = %22.15e\n", F2FRealPart(deriv1));
  printf("deriv, approx  = %22.15e\n", F2FRealPart(deriv_approx));
  printf("relative error = %22.15e\n", F2FRealPart(rel_error1));
  printf("\n");

  // Print out results of test
  printf("V1^{T}*(dQ/dq_{A})^{T}*V2 test with step: %e\n", F2FRealPart(h));
  printf("deriv          = %22.15e\n", F2FRealPart(deriv2));
  printf("deriv, approx  = %22.15e\n", F2FRealPart(deriv_approx));
  printf("relative error = %22.15e\n", F2FRealPart(rel_error2));
  printf("\n");

  // Free allocated memory
  delete[] aero_temps;
  delete[] struct_flux;
  delete[] grad1;
  delete[] grad2;
}
