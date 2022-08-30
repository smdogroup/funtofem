# Top-level Makefile for TransferScheme
include Makefile.in

default:
	@cd src && ${MAKE} || exit 1
	@cd lib && ${MAKE} || exit 1
	@cd funtofem; \
	echo "ctypedef double F2FScalar" > FuntofemTypedefs.pxi; \
	echo "F2F_NPY_SCALAR = np.NPY_DOUBLE" > FuntofemDefs.pxi; \
	echo "dtype = np.double" >> FuntofemDefs.pxi;

debug:
	@cd src && ${MAKE} $@ || exit 1
	@cd lib && ${MAKE} || exit 1
	@cd funtofem; \
	echo "ctypedef double F2FScalar" > FuntofemTypedefs.pxi; \
	echo "F2F_NPY_SCALAR = np.NPY_DOUBLE" > FuntofemDefs.pxi; \
	echo "dtype = np.double" >> FuntofemDefs.pxi;

complex:
	@cd src && ${MAKE} $@ || exit 1
	@cd lib && ${MAKE} || exit 1
	@cd funtofem; \
	echo "ctypedef complex F2FScalar" > FuntofemTypedefs.pxi; \
	echo "F2F_NPY_SCALAR = np.NPY_CDOUBLE" > FuntofemDefs.pxi; \
	echo "dtype = complex" >> FuntofemDefs.pxi;

complex_debug:
	@cd src && ${MAKE} $@ || exit 1
	@cd lib && ${MAKE} || exit 1
	@cd funtofem; \
	echo "ctypedef complex F2FScalar" > FuntofemTypedefs.pxi; \
	echo "F2F_NPY_SCALAR = np.NPY_CDOUBLE" > FuntofemDefs.pxi; \
	echo "dtype = complex" >> FuntofemDefs.pxi;

interface:
	${PIP} install -e .

complex_interface:
	CFLAGS=-DFUNTOFEM_USE_COMPLEX ${PIP} install -e .

clean:
	@cd src && ${MAKE} $@ || exit 1
	@cd lib && ${MAKE} $@ || exit 1
	@rm funtofem/*.so  || exit 1
