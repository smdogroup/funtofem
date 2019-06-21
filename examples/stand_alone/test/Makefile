# Makefile for test_schemes example
include ../../../Makefile.in

default:
	${CXX} ${F2F_INCLUDE} ${CCFLAGS} -o test_schemes.o -c test_schemes.cpp
	${CXX} ${CCFLAGS} -o test_schemes test_schemes.o -L${F2F_DIR}/lib -Wl,-rpath,${F2F_DIR}/lib -ltransfer_schemes ${LAPACK_LIBS}


debug: CCFLAGS=${CCFLAGS_DEBUG}
debug: default

complex: CCFLAGS+=-DFUNTOFEM_USE_COMPLEX
complex: default

complex_debug: CCFLAGS=${CCFLAGS_DEBUG}
complex_debug: CCFLAGS+=-DFUNTOFEM_USE_COMPLEX
complex_debug: default

clean:
	rm test_schemes test_schemes.o
