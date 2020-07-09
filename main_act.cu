#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "cublas_v2.h"
#include "local_lib.h"

#define M 4
#define K 4
#define IDX2C(i,j,ld) (((i)*(ld))+(j))

int main(int argc, char ** argv){
  cudaError_t cudaStat; // cudaMalloc status
  cublasStatus_t stat; // CUBLAS functions status
  cublasHandle_t handle; // CUBLAS context

  float *a; // mxk matrix a on the host
  a = (float*) malloc (M*K* sizeof(float)); // host memory for a

  int i,j,ind = 1;
  for(i=0;i<M;i++){ // 11 ,17 ,23 ,29 ,35
    for(j=0;j<K;j++){ // 12 ,18 ,24 ,30 ,36
      a[IDX2C(i,j,K)]=(float)ind++; // 13 ,19 ,25 ,31 ,37
    } // 14 ,20 ,26 ,32 ,38
  } // 15 ,21 ,27 ,33 ,39

  printf ("a:\n");
  for (i=0;i<M;i ++){
    for (j=0;j<K;j ++){
      printf (" %.5f",a[ IDX2C(i,j,K)]);
    }
  printf ("\n");
  }

  float * d_a; // d_a - a on the device
  cudaStat = cudaMalloc (( void **)& d_a ,M*K* sizeof (float)); // device
  stat = cublasCreate (&handle); // initialize CUBLAS context
  stat = cublasSetMatrix(M, K, sizeof(float), a, M, d_a, M); //a -> d_a
  activationFunction(d_a, M, K);
  stat = cublasGetMatrix (M,K, sizeof (float) ,d_a ,M,a,M); // cp d_c - >c

  printf ("a:\n");
  for (i=0;i<M;i ++){
    for (j=0;j<K;j ++){
      printf (" %.5f",a[ IDX2C(i,j,K)]);
    }
  printf ("\n");
  }
}
