[![Build, unit tests, and docs](https://github.com/smdogroup/funtofem/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/smdogroup/tacs/actions/workflows/unit_tests.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# FUNtoFEM #

This repository contains the FUNtoFEM coupling framework and load and displacement transfer schemes for aeroelastic analysis and optimization.

### Documentation ###

[Documentation for users and developers](https://smdogroup.github.io/funtofem/index.html)

### Installing FUNtoFEM for Users ###
For those intending to be users of funtofem and not developers, the easiest way to install FUNtoFEM is by installing our anaconda package.
Conda packages of funtofem are available for the Linux and Mac OS from our smdogroup channel. The user should first open a terminal and create
a conda environment, such as `F2F`, and then install funtofem as follows with conda install.
```
conda create -n F2F python=3.8
conda activate F2F
conda install -c conda-forge -c smdogroup funtofem
```

All dependencies including linear algebra libraries, and our group's FEA software [TACS](https://github.com/smdogroup/tacs) will be automatically
installed including C++ and Python libraries of funtofem. Optional dependencies such as [OpenMDAO](https://github.com/OpenMDAO/OpenMDAO), [ESP/CAPS](https://acdl.mit.edu/ESP/) will need to be installed by the user.

### License ###

FUNtoFEM is licensed under the Apache License, Version 2.0 (the "License");
you may not use this software except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

### Authors ###

* Jan Kiviaho (jfk115@msstate.edu)
* Kevin Jacobson (kevin.e.jacobson@nasa.gov)
* Graeme Kennedy (graeme.kennedy@aerospace.gatech.edu)
* Liam Smith
* Lenard Halim (lenard.halim@gatech.edu)
* Sejal Sahu (ssahu32@gatech.edu)
* Brian Burke
* Rohan Patel
* Sean Engelstad

### Installing FUNtoFEM for Developers ###
* Dependencies include: MPI, mpi4py, Cython, Lapack/BLAS
* In the funtofem/ directory, copy Makefile.in.info to Makefile.in and edit
* For real mode: `make` and `make interface` (Python interface) in the funtofem/ directory
* For complex mode: `make complex` and `make complex_interface` in the funtofem/ directory
