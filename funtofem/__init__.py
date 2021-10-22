# For new versions of python, DLLs must be loaded via a specific command
import platform, os, sys
if platform.system() == 'Windows' and sys.version_info.minor >= 8:
    # Add Dependencies
    lookfor = ['mkl', 'mpi']
    for k in lookfor:
        for v in os.environ['path'].split(';'):
            if k in v.lower():
                os.add_dll_directory(v)
                print('Adding ', v, ' to DLL search path')
                break
    # Add TransferScheme.dll
    libdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'lib')
    print('Adding ', libdir, ' to DLL search path')
    os.add_dll_directory(libdir)
