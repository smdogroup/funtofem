/* Author:  Lisandro Dalcin   */
/* Contact: dalcinl@gmail.com */

#ifndef MPI_COMPAT_H
#define MPI_COMPAT_H

#include <mpi.h>

// for windows https://githubmemory.com/repo/mpi4py/mpi4py/issues/19 (Adrien Crovato)
#ifdef _WIN32
#define PyMPI_HAVE_MPI_Message 1
#else
// for unix
#if (MPI_VERSION < 3) && !defined(PyMPI_HAVE_MPI_Message)
typedef void *PyMPI_MPI_Message;
#define MPI_Message PyMPI_MPI_Message
#endif
#endif

#endif/*MPI_COMPAT_H*/
