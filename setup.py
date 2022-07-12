import os, sys, platform
from subprocess import check_output

# Numpy/mpi4py must be installed prior to installing FUNtoFEM
import numpy
import mpi4py

# Import distutils
from setuptools import setup
from setuptools.command.build_ext import build_ext
from distutils.core import Extension as Ext
from Cython.Build import cythonize

# Convert from local to absolute directories
def get_global_dir(files):
    funtofem_root = os.path.abspath(os.path.dirname(__file__))
    new = []
    for f in files:
        new.append(os.path.join(funtofem_root, f))
    return new

def get_mpi_flags():
    inc_dirs, lib_dirs, libs = [], [], []
    # Linux/Mac -> openMPI
    if platform.system() == 'Linux' or platform.system() == 'Darwin':
        # Split the output from the mpicxx command
        args = check_output(['mpicxx', '-show']).decode('utf-8').split()
        # Determine whether the output is an include/link/lib command
        for flag in args:
            if flag[:2] == '-I':
                inc_dirs.append(flag[2:])
            elif flag[:2] == '-L':
                lib_dirs.append(flag[2:])
            elif flag[:2] == '-l':
                libs.append(flag[2:])
    # Windows -> MS-MPI
    elif platform.system() == 'Windows':
        inc_dirs.append(os.environ['MSMPI_INC'].rstrip('\\')) # removes trainling "\"
        lib_dirs.append(os.environ['MSMPI_LIB64'].rstrip('\\')) # assuming x64 architecture
        libs.extend(['msmpi', 'msmpifec', 'msmpifmc'])
    else:
        raise Exception('Unsupported OS!\n')

    return inc_dirs, lib_dirs, libs

class custom_build_ext(build_ext):
    def build_extensions(self):
        if platform.system() == 'Windows':
            # Apparently, this is not needed anymore...
            #self.compiler.initialize()
            #self.compiler.cc = '"' + self.compiler.cc + '"' # add double quotes around compiler name
            #self.compiler.linker = '"' + self.compiler.linker + '"' # add double quotes around compiler name
            build_ext.build_extensions(self)
        else:
            build_ext.build_extensions(self)

# Get include dirs and libs
inc_dirs, lib_dirs, libs = get_mpi_flags()

# Add funtofem-dev/lib...
# ... as a runtime directory for linux/mac
if platform.system() == 'Linux' or platform.system() == 'Darwin':
    runtime_lib_dirs = get_global_dir(['lib'])
# ... not for windows, this will be taken care of at runtime (see funtofem/__init__.py)
else:
    runtime_lib_dirs = None

# Relative paths for the include/library directories
rel_inc_dirs = ['src', 'include']
rel_lib_dirs = ['lib']
libs.extend(['transfer_schemes'])

# Convert from relative to absolute directories
inc_dirs.extend(get_global_dir(rel_inc_dirs))
lib_dirs.extend(get_global_dir(rel_lib_dirs))

# Add the numpy/mpi4py directories
inc_dirs.extend([numpy.get_include(), mpi4py.get_include()])

exts = []
for mod in ['TransferScheme']:
    exts.append(Ext('funtofem.%s'%(mod), sources=['funtofem/%s.pyx'%(mod)],
                    include_dirs=inc_dirs, libraries=libs,
                    library_dirs=lib_dirs, runtime_library_dirs=runtime_lib_dirs))

for e in exts:
    e.cython_directives = {"embedsignature": True,
                           "binding":True}

setup(name='funtofem',
      version=0.1,
      description='Aeroelastic coupling framework and transfer schemes',
      author='Graeme J. Kennedy',
      author_email='graeme.kennedy@ae.gatech.edu',
      ext_modules=cythonize(exts, include_path=inc_dirs),
      cmdclass={"build_ext": custom_build_ext},
      packages=['funtofem'],
      package_data={'funtofem': ['__init__.py', 'mphys/*.py']})
