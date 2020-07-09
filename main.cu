#include <assert.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <stdio.h>
#include <stdlib.h>

#include "local_lib.h"

#define M 5
#define K 4
#define IDX2C(i,j,ld) (((i)*(ld))+(j))

int main(int argc, char ** argv){
  cudaError_t cudaStat; // cudaMalloc cublasStatus
  cublasStatus_t cublasStat; // CUBLAS functions cublasStatus
  cublasHandle_t cublasH; // CUBLAS context
  cusolverDnHandle_t cusolverH;
  cusolverStatus_t cusolverStat;

  cublasStat = cublasCreate(&cublasH); // initialize CUBLAS context
  cusolverStat = cusolverDnCreate(&cusolverH);
  assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

  float *a, *aInv; // mxk matrix a on the host
  a = (float*) malloc (M*K* sizeof(float)); // host memory for a
  aInv = (float*) malloc (M*K* sizeof(float));

  int i,j,ind = 1;
  for(i=0;i<M;i++){ // 11 ,17 ,23 ,29 ,35
    for(j=0;j<K;j++){ // 12 ,18 ,24 ,30 ,36
      a[IDX2C(i,j,K)]=(float)ind++; // 13 ,19 ,25 ,31 ,37
    } // 14 ,20 ,26 ,32 ,38
  } // 15 ,21 ,27 ,33 ,39

  printf ("a:\n");
  for (i=0;i<M;i ++){
    for (j=0;j<K;j ++){
      printf (" %.8f",a[ IDX2C(i,j,K)]);
    }
  printf ("\n");
  }

  float * d_a; // d_a - a on the device
  cudaStat = cudaMalloc (( void **)& d_a ,M*K* sizeof (float)); // device
  cublasStat = cublasSetMatrix(M, K, sizeof(float), a, M, d_a, M); //a -> d_a
  float * d_Ainv = getPseudoInverse(cublasH, cusolverH, d_a, M, K);
  cublasStat = cublasGetMatrix (K,M, sizeof (float) ,d_Ainv , K, aInv, K); // cp d_c - >c

  printf ("a:\n");
  for (i=0;i<K;i ++){
    for (j=0;j<M;j ++){
      printf (" %.8f",aInv[ IDX2C(i,j,M)]);
    }
  printf ("\n");
  }

  cublasStat = cublasDestroy_v2(cublasH);
  cusolverStat = cusolverDnDestroy(cusolverH);

}
