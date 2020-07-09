#ifndef LOC_LIB_H
#define LOC_LIB_H
#include<cublas_v2.h>


void activationFunction(float *d_A, int m, int n);
float* getPseudoInverse(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH,
  float *d_A, int m, int n);
float* getIdentityMatrix(int m, float alfa=1.0f);
float* diagonalizeVector(float* vec, int m, bool inverse=false);
cublasStatus_t matMul(cublasHandle_t cublasH, float *d_A, float *d_B, float *d_result,
  int m, int k, int n, bool isRowMajor=false, float alfa=1.0f, float beta=1.0f);

#endif
