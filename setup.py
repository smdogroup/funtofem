import os
from subprocess import check_output
import sys

# Numpy/mpi4py must be installed prior to installing FUNtoFEM
import numpy
import mpi4py

# Import distutils
from setuptools import setup, find_packages
from distutils.core import Extension as Ext
from Cython.Build import cythonize
from Cython.Compiler import Options

Options.docstrings = True

# Convert from local to absolute directories
def get_global_dir(files):
    funtofem_root = os.path.abspath(os.path.dirname(__file__))
    new = []
    for f in files:
        new.append(os.path.join(funtofem_root, f))
    return new


def get_mpi_flags():
    # Split the output from the mpicxx command
    args = check_output(["mpicxx", "-show"]).decode("utf-8").split()

    # Determine whether the output is an include/link/lib command
    inc_dirs, lib_dirs, libs = [], [], []
    for flag in args:
        if flag[:2] == "-I":
            inc_dirs.append(flag[2:])
        elif flag[:2] == "-L":
            lib_dirs.append(flag[2:])
        elif flag[:2] == "-l":
            libs.append(flag[2:])

    return inc_dirs, lib_dirs, libs


inc_dirs, lib_dirs, libs = get_mpi_flags()

# Add funtofem-dev/lib as a runtime directory
runtime_lib_dirs = get_global_dir(["lib"])

# Relative paths for the include/library directories
rel_inc_dirs = ["src", "include"]
rel_lib_dirs = ["lib"]
libs.extend(["transfer_schemes"])

# Convert from relative to absolute directories
inc_dirs.extend(get_global_dir(rel_inc_dirs))
lib_dirs.extend(get_global_dir(rel_lib_dirs))

# Add the numpy/mpi4py directories
inc_dirs.extend([numpy.get_include(), mpi4py.get_include()])

exts = []
for mod in ["TransferScheme"]:
    exts.append(
        Ext(
            "funtofem.%s" % (mod),
            sources=["funtofem/%s.pyx" % (mod)],
            include_dirs=inc_dirs,
            libraries=libs,
            library_dirs=lib_dirs,
            runtime_library_dirs=runtime_lib_dirs,
        )
    )

for e in exts:
    e.cython_directives = {"embedsignature": True, "binding": True}

setup(
    name="funtofem",
    version=0.1,
    description="Aeroelastic coupling framework and transfer schemes",
    author="Graeme J. Kennedy",
    author_email="graeme.kennedy@ae.gatech.edu",
    packages=find_packages(include=["pyfuntofem*"]),
    ext_modules=cythonize(exts, include_path=inc_dirs),
)
