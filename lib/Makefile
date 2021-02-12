# Makefile for TransferScheme library
include ../Makefile.in

F2F_OBJ = ${F2F_DIR}/src/*.o

default:
	${CXX} ${SO_LINK_FLAGS} -o libtransfer_schemes.${SO_EXT} ${F2F_OBJ} ${LAPACK_LIBS}

clean:
	${RM} libtransfer_schemes.${SO_EXT}
	${RM} libtransfer_schemes.a


