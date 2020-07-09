#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define IDX2C(i,j,ld) (((i)*(ld))+(j))
#define M 6 // a - mxk matrix
#define N 4 // b - kxn matrix
#define K 5 // c - mxn matrix

cublasStatus_t matMulRowMajor(cublasHandle_t handle, float *d_A, float *d_B, float *d_result,
  int m, int k, int n){
  /*
    inputs are cublas matrix but row major
    compute d_result^T = d_B^T * d_A^T using cublasSgemm
    d_A (mxk)
    d_B (kxn)
    d_result (mxn)
  */
  float alfa =1.0f;
  float beta =1.0f;
  cublasStatus_t stat=cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,m,k,&alfa,
    d_B,n,d_A,k,&beta,d_result,n);
  return stat;
}

int main ( void ){
  cudaError_t cudaStat; // cudaMalloc status
  cublasStatus_t stat; // CUBLAS functions status
  cublasHandle_t handle; // CUBLAS context
  int i,j; // i-row index ,j- column index
  float *a; // mxk matrix a on the host
  float *b; // kxn matrix b on the host
  float *c; // mxn matrix c on the host
  a= (float*) malloc (M*K* sizeof(float)); // host memory for a
  b= (float*) malloc (K*N* sizeof(float)); // host memory for b
  c= (float*) malloc (M*N* sizeof(float)); // host memory for c

  // define an mxk matrix a column by column
  int ind =11; // a:
  for(i=0;i<M;i++){ // 11 ,17 ,23 ,29 ,35
    for(j=0;j<K;j++){ // 12 ,18 ,24 ,30 ,36
      a[IDX2C(i,j,K)]=(float)ind++; // 13 ,19 ,25 ,31 ,37
    } // 14 ,20 ,26 ,32 ,38
  } // 15 ,21 ,27 ,33 ,39
  // 16 ,22 ,28 ,34 ,40
  // print a row by row
  printf ("a:\n");
  for (i=0;i<M;i ++){
    for (j=0;j<K;j ++){
      printf (" %5.0f",a[ IDX2C(i,j,K)]);
    }
  printf ("\n");
  }
  //
  ind =11; // b:
  for(i=0;i<K;i ++){ // 12 ,17 ,22 ,27
    for(j=0;j<N;j ++){ // 11 ,16 ,21 ,26
      b[ IDX2C (i,j,N )]=( float )ind++; // 13 ,18 ,23 ,28
    } // 14 ,19 ,24 ,29
  } // 15 ,20 ,25 ,30
  // // print b row by row
  printf ("b:\n");
  for (i=0;i<K;i ++){
    for (j=0;j<N;j ++){
      printf (" %5.0f",b[ IDX2C(i,j,N)]);
    }
    printf ("\n");
  }

  for(i=0;i<M;i ++){
    for(j=0;j<N;j ++){
      c[ IDX2C(i,j,N)]=0;
    }
  }
  // // print c row by row
  printf ("c:\n");
  for (i=0;i<M;i ++){
    for (j=0;j<N;j ++){
      printf (" %.5f",c[ IDX2C (i,j,N)]);
    }
    printf ("\n");
  }
  // // on the device
  float * d_a; // d_a - a on the device
  float * d_b; // d_b - b on the device
  float * d_c; // d_c - c on the device
  cudaStat = cudaMalloc (( void **)& d_a ,M*K* sizeof (float)); // device
  // memory alloc for a
  cudaStat = cudaMalloc (( void **)& d_b ,K*N* sizeof (float)); // device
  // memory alloc for b
  cudaStat = cudaMalloc (( void **)& d_c ,M*N* sizeof (float)); // device
  // memory alloc for c
  stat = cublasCreate (&handle); // initialize CUBLAS context
  // // copy matrices from the host to the device
  stat = cublasSetMatrix(M, K, sizeof(float), a, M, d_a, M); //a -> d_a
  stat = cublasSetMatrix(K, N, sizeof(float), b, K, d_b, K); //b -> d_b
  stat = cublasSetMatrix(M, N, sizeof(float), c, M, d_c, M); //c -> d_c
  // // matrix - matrix multiplication : d_c = al*d_a *d_b + bet *d_c
  // // d_a -mxk matrix , d_b -kxn matrix , d_c -mxn matrix ;
  // // al ,bet -scalars
  // // define a kxn matrix b column by column
  stat = matMulRowMajor(handle, d_a, d_b, d_c, M, K, N);
  stat = cublasGetMatrix (M,N, sizeof (*c), d_c, M,c,M); // cp d_c - >c
  printf ("c after Sgemm :\n");
  for(i=0;i<M;i ++){
    for(j=0;j<N;j ++){
      printf (" %.5f",c[IDX2C(i,j,N)]); // print c after Sgemm
    }
    printf ("\n");
  }
  cudaFree (d_a ); // free device memory
  cudaFree (d_b ); // free device memory
  cudaFree (d_c ); // free device memory
  cublasDestroy ( handle ); // destroy CUBLAS context
  free (a); // free host memory
  free (b); // free host memory
  free (c); // free host memory
  return EXIT_SUCCESS ;
}
