# FUNtoFEM #

This repository contains the FUNtoFEM coupling framework and load and displacement transfer schemes for aeroelastic analysis and optimization.

### Documentation ###

[Documentation for users and developers](https://smdogroup.github.io/funtofem/index.html)

### Installation ###

#### Dependencies and options
Dependencies:
* CMake
* MPI and mpi4py
* Cython
* Lapack/BLAS

Options:
* USE_COMPLEX: whether to compile with complex numbers
* USE_MKL: whether to look for Intel MKL instead of openBLAS

#### UNIX (GCC-openMPI)
In the funtofem directory,
```sh
# Configure and build
mkdir build && cd build
cmake [-DCMAKE_BUILD_TYPE=Release|Debug] [-DUSE_COMPLEX=ON|OFF] [-DUSE_MKL=ON|OFF] ..
make install
ctest
```

#### Windows (MSVC-MSMPI)
In the funtofem directory,
```bat
REM Setup env
call "C:\Program Files\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.dat" amd64 REM VS build tools
call "C:\Program Files\Intel\oneAPI\setvars.bat" intel64 vs2019 REM if Intel MKL are used
REM Configure and build
mkdir build
cd build
cmake -A x64 [-DUSE_COMPLEX=ON|OFF] [-DUSE_MKL=ON|OFF] ..
cmake --build . --target install --config Release|Debug
ctest -C Release|Debug
```

#### Python interface
To build the python interface, from the funtofem directory,
```sh
# Build the python interface for the first time
python setup.py develop --user [--define FUNTOFEM_USE_COMPLEX]
# Update the python interface after it has been installed once, if changes are made
python setup.py build_ext --inplace [--define FUNTOFEM_USE_COMPLEX]
```

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
