#ifndef FUNTOFEM_LAPACK_H_
#define FUNTOFEM_LAPACK_H_

/*
  This file contains the definitions of several LAPACK/BLAS functions.
*/

#include "TransferScheme.h"

#define LAPACKsyevd dsyevd_
#define LAPACKdgesvd dgesvd_
#define LAPACKzgesvd zgesvd_

#ifdef FUNTOFEM_USE_COMPLEX
#define LAPACKgetrf zgetrf_
#define LAPACKgetrs zgetrs_
#define BLASgemv zgemv_
#define BLASgemm zgemm_
#else
#define LAPACKgetrf dgetrf_
#define LAPACKgetrs dgetrs_
#define BLASgemv dgemv_
#define BLASgemm dgemm_
#endif

extern "C" {
// Compute an LU factorization of a matrix
extern void LAPACKgetrf(int *m, int *n, F2FScalar *a, int *lda, int *ipiv,
                        int *info);

// This routine solves a system of equations with a factored matrix
extern void LAPACKgetrs(const char *c, int *n, int *nrhs, const F2FScalar *a,
                        int *lda, const int *ipiv, F2FScalar *b, int *ldb,
                        int *info);

// Compute the eigenvalues of a symmetric matrix
extern void LAPACKsyevd(const char *jobz, const char *uplo, int *N, F2FReal *A,
                        int *lda, F2FReal *w, F2FReal *work, int *lwork,
                        int *iwork, int *liwork, int *info);

// Compute the SVD decomposition of a matrix
extern void LAPACKdgesvd(const char *jobu, const char *jobvt, int *m, int *n,
                         F2FReal *a, int *lda, F2FReal *s, F2FReal *u, int *ldu,
                         F2FReal *vt, int *ldvt, F2FReal *work, int *lwork,
                         int *info);

// Compute the complex SVD decomposition of a matrix
extern void LAPACKzgesvd(const char *jobu, const char *jobvt, int *m, int *n,
                         F2FComplex *a, int *lda, F2FReal *s, F2FComplex *u,
                         int *ldu, F2FComplex *vt, int *ldvt, F2FComplex *work,
                         int *lwork, F2FReal *rwork, int *info);

// Level 2 BLAS routines
// y = alpha * A * x + beta * y, for a general matrix
extern void BLASgemv(const char *c, int *m, int *n, F2FScalar *alpha,
                     F2FScalar *a, int *lda, F2FScalar *x, int *incx,
                     F2FScalar *beta, F2FScalar *y, int *incy);

// Level 3 BLAS routines
// C := alpha*op( A )*op( B ) + beta*C,
extern void BLASgemm(const char *ta, const char *tb, int *m, int *n, int *k,
                     F2FScalar *alpha, F2FScalar *a, int *lda, F2FScalar *b,
                     int *ldb, F2FScalar *beta, F2FScalar *c, int *ldc);
}

#endif
