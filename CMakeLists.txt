# ----------------------------------------------------------------------------
PROJECT(FUNtoFEM)
# ----------------------------------------------------------------------------
CMAKE_MINIMUM_REQUIRED(VERSION 3.9)

# -- OPTIONS
OPTION(USE_COMPLEX  "Compile with complex numbers"  OFF)
OPTION(USE_MKL      "Compile with Intel MKL"        ON) # OFF to use openBLAS

# -- I/O
# Build type
IF(NOT CMAKE_BUILD_TYPE)
    SET( CMAKE_BUILD_TYPE "Release" CACHE STRING 
         "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
         FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

# -- C/C++
# Standard
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
# Compile/Link flags
IF(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    # Compiler flags
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -Wpedantic")
    # Linker flags
    IF(NOT APPLE)
        SET(CMAKE_SHARED_LINKER_FLAGS "-fPIC -shared")
    ELSE()
        SET(CMAKE_SHARED_LINKER_FLAGS "-fPIC -dynamiclib -single_module -undefined dynamic_lookup -multiply_defined suppress")
    ENDIF()
ELSEIF(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
	# Compiler flags
	ADD_DEFINITIONS(-D_USE_MATH_DEFINES) # otherwise M_PI is undefined
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /permissive-") # parallel build, recognize alternative keywords (or, ...)
ENDIF()

# -- OUTPUT
# Define lib name and, sources/headers location
FILE(GLOB SRCS src/*.cpp)
ADD_LIBRARY(transfer_schemes SHARED ${SRCS})
TARGET_INCLUDE_DIRECTORIES(transfer_schemes PUBLIC ${PROJECT_SOURCE_DIR}/include)
INSTALL(TARGETS transfer_schemes DESTINATION ${PROJECT_SOURCE_DIR}/lib)
#SET_TARGET_PROPERTIES(transfer_schemes PROPERTIES DEBUG_POSTFIX "_d") # to make the difference between debug and release libs
# Define exe for testing
ADD_EXECUTABLE(test_schemes examples/functionality/stand_alone/test/test_schemes.cpp)
TARGET_LINK_LIBRARIES(test_schemes transfer_schemes)
ENABLE_TESTING()
ADD_TEST(NAME test_schemes COMMAND test_schemes)

# Add definitions
IF(${USE_COMPLEX})
    # C++
    TARGET_COMPILE_DEFINITIONS(transfer_schemes PUBLIC FUNTOFEM_USE_COMPLEX)
    # python
    SET(F2F_TYPE "complex")
    SET(F2F_NPY_TYPE "NPY_CDOUBLE")
ELSE()
    SET(F2F_TYPE "double")
    SET(F2F_NPY_TYPE "NPY_DOUBLE")
ENDIF()
CONFIGURE_FILE("${PROJECT_SOURCE_DIR}/funtofem/FuntofemTypedefs.pxi.in" "${PROJECT_SOURCE_DIR}/funtofem/FuntofemTypedefs.pxi")
CONFIGURE_FILE("${PROJECT_SOURCE_DIR}/funtofem/FuntofemDefs.pxi.in" "${PROJECT_SOURCE_DIR}/funtofem/FuntofemDefs.pxi")

# -- DEPENDENCIES
# MPI
FIND_PACKAGE(MPI REQUIRED)
TARGET_LINK_LIBRARIES(transfer_schemes MPI::MPI_CXX)

# BLAS/LAPACK
# Find MKL
IF(USE_MKL)
    # header (mkl.h) searched using INCLUDE or MKLROOT
    FIND_PATH(MKL_INCLUDE_DIRS NAMES "mkl.h" PATHS "$ENV{MKLROOT}/include")
    MESSAGE(STATUS "MKL_INCLUDE_DIRS=${MKL_INCLUDE_DIRS}")
    # library (mkl_rt.so) searched using LIBRARY_PATH (Linux/macOS) or LIB (windows)
    FIND_LIBRARY(MKL_LIBRARIES mkl_rt PATHS ENV LIBRARY_PATH)
    MESSAGE(STATUS "MKL_LIBRARIES=${MKL_LIBRARIES}")
    IF(MKL_INCLUDE_DIRS AND MKL_LIBRARIES)
        MESSAGE(STATUS "Found Intel MKL")
        SET(FOUND_MKL TRUE)
        SET(LAPACK_INCLUDE_DIRS ${MKL_INCLUDE_DIRS})
        SET(LAPACK_LIBRARIES ${MKL_LIBRARIES})
    ELSE()
        MESSAGE(FATAL_ERROR "Intel MKL not found!")
    ENDIF()
# Find BLAS
ELSE()
    IF(${BLA_VENDOR} MATCHES "OpenBlas")
       FIND_LIBRARY(LAPACK_LIBRARIES openblas)
    ELSE()
       FIND_PACKAGE(LAPACK REQUIRED) #FIND_LAPACK calls FIND_BLAS
    ENDIF()
    IF(NOT LAPACK_LIBRARIES)
        MESSAGE(FATAL_ERROR "BLAS/LAPACK not found!")
    ENDIF()
    MESSAGE(STATUS "LAPACK_LIBRARIES=${LAPACK_LIBRARIES}")  
    MESSAGE(STATUS "LAPACK_LINKER_FLAGS=${LAPACK_LINKER_FLAGS}")  
    MESSAGE(STATUS "BLA_VENDOR=${BLA_VENDOR}")  
    MESSAGE(STATUS "BLA_STATIC=${BLA_STATIC}")
ENDIF()
TARGET_LINK_LIBRARIES(transfer_schemes ${LAPACK_LIBRARIES})

# -- PRINT
MESSAGE(STATUS "PROJECT: ${CMAKE_PROJECT_NAME}")
MESSAGE(STATUS "* SYSTEM NAME=\"${CMAKE_SYSTEM_NAME}\"")
MESSAGE(STATUS "* CXX COMPILER: ${CMAKE_CXX_COMPILER_ID}")
MESSAGE(STATUS "* CXX STANDARD: ${CMAKE_CXX_STANDARD}")
MESSAGE(STATUS "* BUILD TYPE: ${CMAKE_BUILD_TYPE}")
MESSAGE(STATUS "* MKL SUPPORT: ${USE_MKL}")
MESSAGE(STATUS "* COMPLEX MODE: ${USE_COMPLEX}")

