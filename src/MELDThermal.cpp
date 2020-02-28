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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>

#include "MELDThermal.h"
#include "LocatePoint.h"
#include "funtofemlapack.h"

MELDThermal::MELDThermal( MPI_Comm all, MPI_Comm structure, int _struct_root,
                          MPI_Comm aero, int _aero_root, int symmetry,
                          int num_nearest, F2FScalar beta ){
  // Initialize communicators
  global_comm = all;
  struct_comm = structure;
  aero_comm = aero;
  struct_root = _struct_root;
  aero_root = _aero_root;

  // Initialize aerodynamic data member variables
  Xa = NULL;
  Fa = NULL; // Aerodynamic area-weighted normal thermal flux component (scalar)
  na = 0;

  // Initialize structural data member variables
  Xs = NULL;
  Us = NULL; // Temperature, not displacements
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
  MPI_Comm_rank(global_comm,&rank);
  if (rank == struct_root){
    printf("Transfer scheme [%i]: Creating scheme of type MELDThermal...\n", object_id);
  }
}

MELDThermal::~MELDThermal(){
  // Free the aerostructural connectivity data
  if (global_conn){ delete [] global_conn; }

  // Free the load transfer data
  if (global_W){ delete [] global_W; }

  int rank;
  MPI_Comm_rank(global_comm,&rank);
  if ( rank == struct_root){
    printf("Transfer scheme [%i]: freeing MELD data...\n", object_id);
  }
}

void MELDThermal::setStructNodes( const F2FScalar *struct_X, int struct_nnodes ){
  // Free the structural data if any is allocated
  if (Xs_local){ delete [] Xs_local; Xs_local = NULL; }

  ns_local = struct_nnodes;
  Xs_local = new F2FScalar[3*ns_local];
  memcpy(Xs_local, struct_X, 3*ns_local*sizeof(F2FScalar));

  mesh_update = 1;
}

/*
  Set the aerodynamic surface node locations
*/
void MELDThermal::setAeroNodes(const F2FScalar *aero_X, int aero_nnodes){
  na = aero_nnodes;

  // Free the aerodynamic data if any is allocated
  if (Xa){ delete [] Xa; }
  if (Fa){ delete [] Fa; }

  // Global number of aerodynamic nodes
  na_global = 0;
  MPI_Allreduce(&na, &na_global, 1, MPI_INT, MPI_SUM, global_comm);

  // Allocate memory for aerodynamic data, copy in node locations, initialize
  // displacement and load arrays
  if (na > 0){
    Xa = new F2FScalar[3*na];
    memcpy(Xa, aero_X, 3*na*sizeof(F2FScalar));

    Fa = new F2FScalar[na];
    memset(Fa, 0, 1*na*sizeof(F2FScalar));
  }
}

/*
  Collect a structural vector to create a global image then distribute to the
  aerodynamic processors
*/
void MELDThermal::collectStructuralVector( const F2FScalar *local, F2FScalar *global,
					   int vars_per_node ){
  // Collect how many structural nodes every processor has
  if (struct_comm != MPI_COMM_NULL){
    int struct_nprocs;
    int struct_rank;
    MPI_Comm_size(struct_comm, &struct_nprocs);
    MPI_Comm_rank(struct_comm, &struct_rank);

    int *ns_list = new int[struct_nprocs];
    memset(ns_list, 0, struct_nprocs*sizeof(int));

    MPI_Gather(&ns_local, 1, MPI_INT, ns_list, 1, MPI_INT, struct_root, struct_comm);

    // Collect the structural nodes on the master
    int send_size = ns_local*vars_per_node;
    int *disps = new int[struct_nprocs];
    memset(disps, 0, struct_nprocs*sizeof(int));

    if ( struct_rank == struct_root){
      for (int proc = 0; proc < struct_nprocs; proc++){
        ns_list[proc] *= vars_per_node;
        if (proc > 0){
          disps[proc] = disps[proc-1] + ns_list[proc-1];
        }
      }
    }

    MPI_Gatherv(local, send_size, F2F_MPI_TYPE, global, ns_list, disps, F2F_MPI_TYPE, 0, struct_comm);

    delete [] ns_list;
    delete [] disps;
  }

  // Pass the global list to all the processors
  MPI_Bcast(global, vars_per_node*ns, F2F_MPI_TYPE, struct_root, global_comm);
}

/*
  Reduce vector to get the total across all aero procs then distribute to the
  structural processors
*/
void MELDThermal::distributeStructuralVector( F2FScalar *global, F2FScalar *local,
					      int vars_per_node ){
  // Get the contributions from each aero processor
  MPI_Allreduce(MPI_IN_PLACE, global, ns*vars_per_node, F2F_MPI_TYPE, MPI_SUM, global_comm);

  // Collect how many nodes each structural processor has
  if ( struct_comm != MPI_COMM_NULL ) {
    int struct_nprocs;
    int struct_rank;
    MPI_Comm_size(struct_comm, &struct_nprocs);
    MPI_Comm_rank(struct_comm, &struct_rank);

    int *ns_list = new int[struct_nprocs];

    MPI_Gather(&ns_local,1, MPI_INT, ns_list, 1, MPI_INT, 0, struct_comm);

    // Distribute to the structural processors
    int *disps = new int[struct_nprocs];
    memset(disps, 0, struct_nprocs*sizeof(int));

    if ( struct_rank == 0){
      for (int proc = 0; proc < struct_nprocs; proc++){
        ns_list[proc] *= vars_per_node;
        if (proc > 0){
          disps[proc] = disps[proc-1] + ns_list[proc-1];
        }
      }
    }

    MPI_Scatterv(global, ns_list, disps, F2F_MPI_TYPE, local, ns_local*vars_per_node, F2F_MPI_TYPE, 0, struct_comm);

    delete [] ns_list;
    delete [] disps;
  }
}

void MELDThermal::distributeStructuralMesh(){
  MPI_Allreduce(MPI_IN_PLACE, &mesh_update, 1, MPI_INT, MPI_SUM, global_comm);
  if (mesh_update > 0){
    ns = 0;
    if (struct_comm != MPI_COMM_NULL){
      MPI_Reduce(&ns_local, &ns, 1, MPI_INT, MPI_SUM, 0, struct_comm);
    }

    MPI_Bcast(&ns, 1, MPI_INT, struct_root, global_comm);

    // Allocate memory for structural data, initialize displacement array
    if (Xs){ delete [] Xs; }
    if (Us){ delete [] Us; }

    Xs = new F2FScalar[3*ns];
    memset(Xs, 0, 3*ns*sizeof(F2FScalar));

    Us = new F2FScalar[ns];
    memset(Us, 0, ns*sizeof(F2FScalar));

    collectStructuralVector(Xs_local, Xs);
    if(Xs_local){ delete [] Xs_local; Xs_local = NULL;}
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
  if (nn > ns) { nn = ns; }

  // Create aerostructural connectivity
  global_conn = new int[nn*na];
  setAeroStructConn(global_conn);

  // Allocate and compute the weights
  global_W = new F2FScalar[nn*na];
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
  int *locate_to_reflected_index = NULL;

  if (isymm > -1) {
    Xs_dup = new F2FScalar[6*ns];
    memcpy(Xs_dup, Xs, 3*ns*sizeof(F2FScalar));

    double tol = 1e-7;
    for (int k = 0; k < ns; k++) {
      if (fabs(F2FRealPart(Xs_dup[3*k+isymm])) > tol) {
        // Node is not on the plane of symmetry, so copy and...
        memcpy(&Xs_dup[3*(ns+k)], &Xs_dup[3*k], 3*sizeof(F2FScalar));
        // reflect
        Xs_dup[3*(ns+k)+isymm] *= -1.0;
      } else {
        Xs_dup[3*(ns+k)+0] = -(k+1)*9.0e9;
        Xs_dup[3*(ns+k)+1] = -(k+1)*9.0e9+1;
        Xs_dup[3*(ns+k)+2] = -(k+1)*9.0e9+2;
      }
    }
    num_locate_nodes = 2*ns;
  } else {
    Xs_dup = new F2FScalar[3*ns];
    memcpy(Xs_dup, Xs, 3*ns*sizeof(F2FScalar));
    num_locate_nodes = ns;
  }

  // Create instance of LocatePoint class to perform the following searches
  int min_bin_size = 10;
  // bug is in one of these variables:
  LocatePoint *locator = new LocatePoint(Xs_dup, num_locate_nodes, min_bin_size);

  // Indices of nearest nodes
  int *indx = new int[nn];
  F2FScalar *dist = new F2FScalar[nn];

  // For each aerodynamic node, copy the indices of the nearest n structural
  // nodes into the conn array
  for ( int i = 0; i < na; i++ ) {
    F2FScalar xa0[3];
    memcpy(xa0, &Xa[3*i], 3*sizeof(F2FScalar));

    locator->locateKClosest(nn, indx, dist, xa0);
    memcpy(&conn[nn*i], indx, nn*sizeof(int));
  }

  // Free the duplicate array
  delete [] Xs_dup;

  // Delete the LocatePoint object and release memory
  delete [] indx;
  delete [] dist;
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
    const F2FScalar *xa0 = &Xa[3*i];
    const int *local_conn = &global_conn[i*nn];
    F2FScalar *w = &W[i*nn];

    // Compute the weights based on the difference between the aerodynamic point
    // location and the local undeformed structural point
    F2FScalar wtotal = 0.0;
    F2FScalar v[3];

    // Find the average squared distance for normalization
    F2FScalar v2_avg = 0.0;
    F2FScalar dist;

    for ( int j = 0; j < nn; j++ ){
      if (local_conn[j] < ns) {
        const F2FScalar *xs0 = &Xs[3*local_conn[j]];
        vec_diff(xs0, xa0, v);
      }
      else {
        F2FScalar rxs0[3]; // Reflected xs0
        memcpy(rxs0, &Xs[3*(local_conn[j] - ns)], 3*sizeof(F2FScalar));
        rxs0[isymm] *= -1.0;
        vec_diff(rxs0, xa0, v);
      }
      v2_avg += vec_dot(v, v)/nn;
    }

    // Make sure we don't divide by zero
    if (F2FRealPart(v2_avg) < 1.0e-7){
      v2_avg = 1.0e-7;
    }

    for ( int j = 0; j < nn; j++ ){
      if (local_conn[j] < ns) {
        const F2FScalar *xs0 = &Xs[3*local_conn[j]];
        vec_diff(xs0, xa0, v);
      }
      else {
        F2FScalar rxs0[3]; // Reflected xs0
        memcpy(rxs0, &Xs[3*(local_conn[j] - ns)], 3*sizeof(F2FScalar));
        rxs0[isymm] *= -1.0;
        vec_diff(rxs0, xa0, v);
      }
      w[j] = exp(-global_beta*vec_dot(v, v)/v2_avg);
      wtotal += w[j];
    }

    // Normalize the weights
    wtotal = 1.0/wtotal;
    for ( int j = 0; j < nn; j++ ){
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
  collectStructuralVector(struct_Temp, Us, 1); // set vars_per_node = 1 for temps

  // Zero the outputs
  memset(aero_Temp, 0.0, na*sizeof(F2FScalar));

  for ( int i = 0; i < na; i++ ) {
    const int *local_conn = &global_conn[i*nn];
    const F2FScalar *w = &global_W[i*nn];

    F2FScalar Taero = 0.0;
    for ( int j = 0; j < nn; j++ ){
      if (local_conn[j] < ns) {
        Taero += w[j]*Us[local_conn[j]];
      }
      else {
        Taero += w[j]*Us[local_conn[j] -ns];
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
  memcpy(Fa, aero_flux, na*sizeof(F2FScalar));  

  // Zero struct flux
  F2FScalar *struct_flux_global = new F2FScalar[ns];
  memset(struct_flux_global, 0, ns*sizeof(F2FScalar));

  for ( int i = 0; i < na; i++ ) {
    const int *local_conn = &global_conn[i*nn];
    const F2FScalar *w = &global_W[i*nn];
    const F2FScalar *fa = &Fa[i];

    for ( int j = 0; j < nn; j++ ){
      int index = 0;
      if (local_conn[j] < ns) {
        index = local_conn[j];
      }
      else {
        index = local_conn[j] - ns;
      }
      struct_flux_global[index] += w[j]*fa[0];
    }
  }

  // distribute the structural loads
  distributeStructuralVector(struct_flux_global, struct_flux, 1); // set vars_per_node = 1 for flux
  delete [] struct_flux_global;
}
