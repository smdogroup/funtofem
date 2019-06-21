include ${HOME}/hg/tacs/Makefile.in
include ${HOME}/hg/tacs/TACS_Common.mk
include ../../Makefile.in
include ../../FUNtoFEM_Common.mk

OBJS = crm.o

# New rule for combined TransferScheme/TACS .c files
%.o: %.c
	${CXX} ${TACS_CC_FLAGS} ${F2F_CC_FLAGS} -c $< -o $*.o

default: ${OBJS}
	${CXX} -o crm ${OBJS} ${TACS_LD_FLAGS} ${F2F_LD_FLAGS}

debug: TACS_CC_FLAGS=${TACS_DEBUG_CC_FLAGS}
debug: F2F_CC_FLAGS=${F2F_DEBUG_CC_FLAGS}
debug: default

complex: CCFLAGS+=-DF2F_USE_COMPLEX
complex: TACS_DEF+=-DTACS_USE_COMPLEX
complex: default

clean:
	rm -rf ${OBJS} crm
