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
#include "TACSMeshLoader.h"
#include "MITCShell.h"
#include "isoFSDTStiffness.h"
#include "TransferScheme.h"
#include "MELD.h"

int main( int argc, char *argv[] ){
  // Initialize MPI and declare communicator
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;

/*   
  In TACS
  -----------
  Load in the structural mesh, create TACS from the structural mesh, assemble
  the stiffness matrix, and load in the aerodynamic surface mesh
*/

  // Load structural mesh from BDF file using TACS MeshLoader
  const char *filename = "../resources/CRM_box_2nd.bdf";
  TACSMeshLoader *mesh = new TACSMeshLoader(comm);
  mesh->incref();
  mesh->scanBDFFile(filename);

  // Get number of components specified in BDF file
  int num_components = mesh->getNumComponents();

  // Set properties needed to create constitutive object 
  double rho = 2500.0; // density, kg/m^3
  double E = 70e9; // elastic modulus, Pa
  double nu = 0.3; // poisson's ratio
  double kcorr = 5.0/6.0; // shear correction factor
  double ys = 350e6; // yield stress, Pa

  // Loop over components, creating constituitive object for each
  for ( int i = 0; i < num_components; i++ ){
    const char *descriptor = mesh->getElementDescript(i);
    double min_thickness = 0.01;
    double max_thickness = 0.20;
    double thickness = 0.07;
    isoFSDTStiffness *stiff = new isoFSDTStiffness(rho, E, nu, kcorr, ys,
        thickness, i, min_thickness, max_thickness); 

    // Initialize element object
    TACSElement *element = NULL;

    // Create element object using constituitive information and type defined in
    // descriptor
    if ( strcmp(descriptor, "CQUAD") == 0 || 
         strcmp(descriptor, "CQUADR") == 0 ||
         strcmp(descriptor, "CQUAD4") == 0) {
      element = new MITCShell<2>(stiff, LINEAR, i);
    }
    mesh->setElement(i, element);
  }

  // Create TACS Assembler from TACS MeshLoader
  TACSAssembler *tacs = mesh->createTACS(6);
  tacs->incref();
  mesh->decref();

  // Create matrix and vectors 
  TACSBVec *ans = tacs->createVec(); // displacements and rotations
  TACSBVec *f = tacs->createVec(); // loads
  FEMat *mat = tacs->createFEMat(); // preconditioner
  TACSBVec *struct_X_vec = tacs->createNodeVec();

  // Increment reference count to the matrix/vectors
  ans->incref();
  f->incref();
  mat->incref();
  struct_X_vec->incref();

  // Allocate the factorization
  int lev = 10000;
  double fill = 10.0;
  int reorder_schur = 1;
  PcScMat *pc = new PcScMat(mat, lev, fill, reorder_schur); 
  pc->incref();

  // Assemble and factor the stiffness matrix
  double alpha = 1.0, beta = 0.0, gamma = 0.0;
  tacs->assembleJacobian(alpha, beta, gamma, NULL, mat);
  tacs->applyBCs(mat);
  pc->factor();

  // Retrieve node locations and number of nodes for structual mesh
  int struct_nnodes = tacs->getNumNodes();
  tacs->getNodes(struct_X_vec);
  TacsScalar *struct_X;
  int struct_size = struct_X_vec->getArray(&struct_X);

  // Load aerodynamic mesh using TACS MeshLoader
  const char *aero_filename = "../resources/ucrm_aero_mesh.bdf";
  TACSMeshLoader *aero_mesh = new TACSMeshLoader(comm);
  aero_mesh->incref();
  aero_mesh->scanBDFFile(aero_filename);

  // Get number of components prescribed in BDF file
  int aero_num_components = aero_mesh->getNumComponents();

  // Loop over components, creating constituitive object for each
  for ( int i = 0; i < aero_num_components; i++ ){
    const char *aero_descriptor = aero_mesh->getElementDescript(i);
    double min_thickness = 0.01;
    double max_thickness = 0.20;
    double thickness = 0.07;
    isoFSDTStiffness *aero_stiff = new isoFSDTStiffness(rho, E, nu, kcorr, ys,
        thickness, i, min_thickness, max_thickness); 

    // Initialize element object
    TACSElement *element = NULL;

    // Create element object using constituitive information and type defined in
    // descriptor
    if ( strcmp(aero_descriptor, "CQUAD") == 0 || 
         strcmp(aero_descriptor, "CQUADR") == 0 ||
         strcmp(aero_descriptor, "CQUAD4") == 0) {
      element = new MITCShell<2>(aero_stiff, LINEAR, i);
    }
    aero_mesh->setElement(i, element);
  }

  // Create tacs assembler from mesh loader object
  TACSAssembler *aero_tacs = aero_mesh->createTACS(6);
  aero_tacs->incref();
  aero_mesh->decref();

  // Retrieve node locations and number of nodes for aero mesh from TACS
  TACSBVec *aero_X_vec = aero_tacs->createNodeVec();
  aero_X_vec->incref();
  int aero_nnodes = aero_tacs->getNumNodes();
  aero_tacs->getNodes(aero_X_vec);
  TacsScalar *aero_X;
  int aero_size = aero_X_vec->getArray(&aero_X);

/*   
  In TransferScheme
  -----------------
  Loading in the meshes, inventing loads, and transferring loads
*/

  // Specify scheme type, symmetry, number of nearest nodes
  int isymm = -1;
  int num_nearest = 100;
  F2FScalar decay_param = 0.5;

  // Create instance of transfer scheme class
  TransferScheme *meld = new MELD(comm, comm, 0, comm, 0, isymm, num_nearest,
                                  decay_param);

  // Initialize transfer scheme
  meld->setStructNodes(struct_X, struct_nnodes);
  meld->setAeroNodes(aero_X, aero_nnodes);
  meld->initialize();

  // Create loads on aero mesh and apply load transfer
  F2FScalar *aero_loads = new F2FScalar[ 3*aero_nnodes ];
  for ( int i = 0 ; i < aero_nnodes; i++ ) {
    aero_loads[3*i+0] = 0.0;
    aero_loads[3*i+1] = 0.0;
    aero_loads[3*i+2] = 10000.0;
  }
  F2FScalar *init_disps = new F2FScalar[3*struct_nnodes];
  memset(init_disps, 0.0, 3*struct_nnodes*sizeof(F2FScalar)); 
  F2FScalar *aero_disps = new F2FScalar[3*aero_nnodes];
  memset(aero_disps, 0.0, 3*aero_nnodes*sizeof(F2FScalar)); 
  F2FScalar *struct_loads = new F2FScalar[3*struct_nnodes];
  memset(struct_loads, 0.0, 3*struct_nnodes*sizeof(F2FScalar)); 

  meld->transferDisps(init_disps, aero_disps);
  meld->transferLoads(aero_loads, struct_loads);

/*   
  In TACS
  -----------
  Setting the loads, solving, retrieving displacements, configuring output
*/

  // Set load vector in TACS to load vector from transfer scheme
  // (For shell elements, first 3 degrees of freedom in residual vector f
  // correspond to forces; last 3 correspond to moments. We will populate the
  // first 3.)
  TacsScalar *f_array;
  f->zeroEntries();
  int f_size = f->getArray(&f_array);
  for ( int i = 0; i < struct_nnodes; i++ ){
    memcpy(&f_array[6*i], &struct_loads[3*i], 3*sizeof(TacsScalar));
  }
  tacs->applyBCs(f);

  // Solve and set solution into ans vector
  pc->applyFactor(f, ans);
  tacs->setVariables(ans);
  
  // Extract displacements from the ans vector
  // (For shell elements, first 3 degrees of freedom correspond to u,v,w
  // displacements; last 3 correspond to rotations)
  TacsScalar *struct_disps = new TacsScalar[ 3*struct_nnodes ];
  memset(struct_disps, 0, 3*struct_nnodes*sizeof(TacsScalar));
  TacsScalar *ans_array;
  ans->getArray(&ans_array);
  for ( int i = 0; i < struct_nnodes; i++ ){
    memcpy(&struct_disps[3*i], &ans_array[6*i], 3*sizeof(TacsScalar));
  }

  // Overwrite rotations in ans vector with forces computing by transfer scheme in
  // order to visualize them in TACS output
  for ( int i = 0; i < struct_nnodes; i++ ){
    memcpy(&ans_array[6*i+3], &f_array[3*i], 3*sizeof(TacsScalar));
  }
  tacs->setVariables(ans);

  // Create an TACSToFH5 object for writing output to files
  unsigned int write_flag = (TACSElement::OUTPUT_NODES |
                             TACSElement::OUTPUT_DISPLACEMENTS |
                             TACSElement::OUTPUT_STRAINS |
                             TACSElement::OUTPUT_STRESSES |
                             TACSElement::OUTPUT_EXTRAS);
  TACSToFH5 * f5 = new TACSToFH5(tacs, TACS_SHELL, write_flag);
  f5->incref();

  // Write the displacements
  f5->writeToFile("ucrm.f5");

/*   
  In transfer scheme
  -----------
  Transfer the displacements computed by TACS and output data file
*/

  // Transfer displacements
  meld->transferDisps(struct_disps, aero_disps);

/*   
  Clean up
  ---------
*/

  // Free all the allocated data
  delete [] init_disps;
  delete [] aero_disps;
  delete [] aero_loads;
  delete [] struct_loads;

  // Call transfer scheme destructor
  delete meld;

  // Decrease the reference count for TACS objects
  pc->decref();
  mat->decref();
  ans->decref();
  f->decref();
  struct_X_vec->decref();
  aero_X_vec->decref();
  f5->decref();
  tacs->decref();
  aero_tacs->decref();

  // Finalize MPI and return
  MPI_Finalize();
  return (0);
}
