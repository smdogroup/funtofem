export F2F_DIR=${SRC_DIR}
# to test the build run the following command from the F2F root folder
# conda build -c conda-forge -c "smdogroup/label/complex" -c smdogroup conda;

# platform specific environment variables
if [[ $(uname) == Darwin ]]; then
  export SO_EXT="dylib"
  export SO_LINK_FLAGS="-fPIC -dynamiclib -headerpad_max_install_names"
  export LIB_SLF="${SO_LINK_FLAGS} -install_name @rpath/libtransfer_schemes.dylib"
  export LAPACK_LIBS="-framework accelerate"
elif [[ "$target_platform" == linux-* ]]; then
  export SO_EXT="so"
  export SO_LINK_FLAGS="-fPIC -shared"
  export LIB_SLF="${SO_LINK_FLAGS}"
  export LAPACK_LIBS="-L${PREFIX}/lib/ -llapack -lpthread -lblas"
fi


if [[ $scalar == "complex" ]]; then
  export OPTIONAL="complex"
  export PIP_FLAGS="-DFUNTOFEM_USE_COMPLEX"
elif [[ $scalar == "real" ]]; then
  export OPTIONAL="default"
fi

# make funtofem and move python object file to conda
cp Makefile.in.info Makefile.in;
make ${OPTIONAL} F2F_DIR=$F2F_DIR;
mv ${F2F_DIR}/lib/* ${PREFIX}/lib;

# copy all header files to conda dir
mkdir ${PREFIX}/include/funtofem
cp ${F2F_DIR}/include/*.h ${PREFIX}/include/funtofem

# make the python package
CPPFLAGS=${PIP_FLAGS} ${PYTHON} -m pip install --no-deps --prefix=${PREFIX} . -vv;