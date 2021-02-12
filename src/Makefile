include ../Makefile.in

F2F_SRC = $(wildcard *.cpp)
F2F_OBJ = $(F2F_SRC:.cpp=.o)

default: ${F2F_OBJ}
	${AR} ${AR_FLAGS} ${F2F_DIR}/lib/libtransfer_schemes.a ${F2F_OBJ}
	
debug: CXXFLAGS=${CXXFLAGS_DEBUG}
debug: default

complex: CXXFLAGS+=-DFUNTOFEM_USE_COMPLEX
complex: default

complex_debug: CXXFLAGS=${CXXFLAGS_DEBUG}
complex_debug: complex

%.o: %.cpp
	${CXX} ${F2F_INCLUDE} ${CXXFLAGS} -c $< -o $@

clean:
	${RM} *.o
