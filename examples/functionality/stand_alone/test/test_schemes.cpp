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
#include <cstdlib>
#include "MELD.h"
#include "LinearizedMELD.h"
#include "RBF.h"

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  // Seed random number generator
  time_t t;
  srand((unsigned) time(&t));

  // Create aerodynamic nodes and displacements
  int aero_nnodes = 30;
  F2FScalar *XA0 = new F2FScalar[3*aero_nnodes];
  F2FScalar *FA = new F2FScalar[3*aero_nnodes];
  for (int i = 0; i < 3*aero_nnodes; i++ ) {
    XA0[i] = (1.0*rand())/RAND_MAX;
    FA[i] = (1.0*rand())/RAND_MAX;
  }

  // Create structural nodes and displacements
  int struct_nnodes = 50;
  F2FScalar *XS0 = new F2FScalar[3*struct_nnodes];
  F2FScalar *US = new F2FScalar[3*struct_nnodes];
  for (int j = 0; j < 3*struct_nnodes; j++ ) {
    XS0[j] = (1.0*rand())/RAND_MAX;
    US[j] = (1.0*rand())/RAND_MAX;
  }
  
  // Create random testing data
  F2FScalar *test_vec_a1 = new F2FScalar[3*aero_nnodes];
  F2FScalar *test_vec_a2 = new F2FScalar[3*aero_nnodes];
  for (int i = 0; i < 3*aero_nnodes; i++) {
    test_vec_a1[i] = (1.0*rand())/RAND_MAX;
    test_vec_a2[i] = (1.0*rand())/RAND_MAX;
  }
  
  F2FScalar *US_pert = new F2FScalar[3*struct_nnodes];
  F2FScalar *test_vec_s1 = new F2FScalar[3*struct_nnodes];
  F2FScalar *test_vec_s2 = new F2FScalar[3*struct_nnodes];
  for (int j = 0; j < 3*struct_nnodes; j++ ) {
    US_pert[j] = (1.0*rand())/RAND_MAX;
    test_vec_s1[j] =  (1.0*rand())/RAND_MAX;
    test_vec_s2[j] =  (1.0*rand())/RAND_MAX;
  }

#ifdef FUNTOFEM_USE_COMPLEX
  F2FScalar h = 1.0e-30;
#else
  F2FScalar h = 1.0e-6;
#endif

  // Create transfer scheme of type MELD
  MPI_Comm comm = MPI_COMM_WORLD;
  int symmetry = 1;
  int nn = 10;
  F2FScalar beta = 0.5;
  TransferScheme *meld = new MELD(comm, comm, 0, comm, 0, symmetry, nn, beta);
  meld->setAeroNodes(XA0, aero_nnodes);
  meld->setStructNodes(XS0, struct_nnodes);
  meld->initialize();

  // Test MELD
  meld->testLoadTransfer(US, FA, US_pert, h);
  meld->testDispJacVecProducts(US, test_vec_a1, test_vec_s1, h);
  meld->testLoadJacVecProducts(US, FA, test_vec_s1, test_vec_s2, h);
  meld->testdDdxA0Products(US, test_vec_a1, test_vec_a2, h);
  meld->testdDdxS0Products(US, test_vec_a1, test_vec_s1, h);
  meld->testdLdxA0Products(US, FA, test_vec_a1, test_vec_s1, h);
  meld->testdLdxS0Products(US, FA, test_vec_s1, test_vec_s2, h);
  delete meld;

  // Create transfer scheme of type linearizedMELD
  //TransferScheme *linmeld = new LinearizedMELD(comm, comm, 0, comm, 0, nn, beta);
  //linmeld->setAeroNodes(XA0, aero_nnodes);
  //linmeld->setStructNodes(XS0, struct_nnodes);
  //linmeld->initialize();

  // Test linearizedMELD
  //linmeld->testLoadTransfer(US, FA, US_pert, h);
  //linmeld->testDispJacVecProducts(US, test_vec_a1, test_vec_s1, h);
  //linmeld->testLoadJacVecProducts(US, FA, test_vec_s1, test_vec_s2, h);
  //linmeld->testdDdxA0Products(US, test_vec_a1, test_vec_a2, h);
  //linmeld->testdDdxS0Products(US, test_vec_a1, test_vec_s1, h);
  //linmeld->testdLdxA0Products(US, FA, test_vec_a1, test_vec_s1, h);
  //linmeld->testdLdxS0Products(US, FA, test_vec_s1, test_vec_s2, h);
  //delete linmeld;

  // Create transfer scheme of type RBF
  RBF::RbfType rbf_type = RBF::THIN_PLATE_SPLINE;
  int denom = 2;
  TransferScheme *rbf = new RBF(comm, comm, 0, comm, 0, rbf_type, denom);
  rbf->setAeroNodes(XA0, aero_nnodes);
  rbf->setStructNodes(XS0, struct_nnodes);
  rbf->initialize();

  // Test RBF
  rbf->testLoadTransfer(US, FA, US_pert, h);
  rbf->testDispJacVecProducts(US, test_vec_a1, test_vec_s1, h);
  rbf->testLoadJacVecProducts(US, FA, test_vec_s1, test_vec_s2, h);
  rbf->testdDdxA0Products(US, test_vec_a1, test_vec_a2, h);
  rbf->testdDdxS0Products(US, test_vec_a1, test_vec_s1, h);
  rbf->testdLdxA0Products(US, FA, test_vec_a1, test_vec_s1, h);
  rbf->testdLdxS0Products(US, FA, test_vec_s1, test_vec_s2, h);
  delete rbf;
  
  // Free allocated memory
  delete [] XA0;
  delete [] FA;
  delete [] XS0;
  delete [] US;
  delete [] US_pert;
  delete [] test_vec_a1;
  delete [] test_vec_a2;
  delete [] test_vec_s1;
  delete [] test_vec_s2;

  MPI_Finalize();
}
