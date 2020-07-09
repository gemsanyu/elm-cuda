#include <assert.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <math.h>
#include <stdio.h>

#include "local_lib.h"

#define BLOCK_SIZE 256
#define GRID_SIZE 50000
#define EPS 10e-5

cublasStatus_t matMul(cublasHandle_t cublasH, float *d_A, float *d_B, float *d_result,
  int m, int k, int n, bool isRowMajor, float alfa, float beta){
  cublasStatus_t stat;
  if (isRowMajor){
    stat=cublasSgemm(cublasH,CUBLAS_OP_N,CUBLAS_OP_N,n,m,k,&alfa,
      d_B,n,d_A,k,&beta,d_result,n);
  } else {
    stat=cublasSgemm(cublasH,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&alfa,
      d_A,m,d_B,k,&beta,d_result,m);
  }
  return stat;
}

__global__ void d_ActivationFunction(float *d_A, int *d_m, int *d_n){
  int m = (*d_m);
  int n = (*d_n);
  int size = m*n;
  int bIdx = blockIdx.x;
  int tIdx = threadIdx.x;
  int stride = blockDim.x;
  int id = bIdx*stride + tIdx;
  for(int i=id;i<size;i+=stride){
    d_A[i] = 1.0 / (1.0 + exp(-d_A[i]));
  }
}

void activationFunction(float *d_A, int m, int n){
  int gridSize = (m*n/BLOCK_SIZE + 1);
  gridSize = min(gridSize, GRID_SIZE);

  int *d_m, *d_n;
  cudaMalloc(&d_m, sizeof(int));
  cudaMalloc(&d_n, sizeof(int));
  cudaMemcpy(d_m, &m, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);
  d_ActivationFunction<<< gridSize, BLOCK_SIZE >>>(d_A, d_m, d_n);
  cudaFree(&d_m);
  cudaFree(&d_n);
}

__global__ void d_GetIdentityMatrix(float *d_A, int *d_m, float *d_Alfa){
  int m = (*d_m);
  float alfa = *d_Alfa;
  int bIdx = blockIdx.x;
  int tIdx = threadIdx.x;
  int blockStride = blockDim.x;
  int gridStride = gridDim.x;
  for(int bid=bIdx;bid<m;bid+=gridStride){
    for(int tid=tIdx;tid<m;tid+=blockStride){
      d_A[bid*m + tid] = 0;
      if(bid==tid){
        d_A[bid*m + tid] += alfa;
      }
    }
  }
}

float* getIdentityMatrix(int m, float alfa){
  int gridSize = (m*m/BLOCK_SIZE + 1);
  gridSize = min(gridSize, GRID_SIZE);

  int *d_m;
  cudaMalloc(&d_m, sizeof(int));
  cudaMemcpy(d_m, &m, sizeof(int), cudaMemcpyHostToDevice);

  float *d_Alfa;
  cudaMalloc(&d_Alfa, sizeof(float));
  cudaMemcpy(d_Alfa, &alfa, sizeof(float), cudaMemcpyHostToDevice);

  float *d_A;
  cudaMalloc(&d_A, m*m*sizeof(float));
  d_GetIdentityMatrix<<< gridSize, BLOCK_SIZE >>>(d_A, d_m, d_Alfa);
  cudaFree(&d_m);
  cudaFree(&d_Alfa);
  return d_A;
}

__global__ void d_GetDiagMatrix(float* d_vec, float* diagMat, int *d_m){
  int m = (*d_m);
  int bIdx = blockIdx.x;
  int tIdx = threadIdx.x;
  int blockStride = blockDim.x;
  int gridStride = gridDim.x;
  for(int bid=bIdx;bid<m;bid+=gridStride){
    for(int tid=tIdx;tid<m;tid+=blockStride){
      diagMat[bid*m + tid] = 0;
      if(bid==tid){
        diagMat[bid*m + tid] = d_vec[tid];
      }
    }
  }
}

__global__ void d_GetInverseDiagMatrix(float* d_vec, float* diagMat, int *d_m){
  int m = (*d_m);
  int bIdx = blockIdx.x;
  int tIdx = threadIdx.x;
  int blockStride = blockDim.x;
  int gridStride = gridDim.x;
  for(int bid=bIdx;bid<m;bid+=gridStride){
    for(int tid=tIdx;tid<m;tid+=blockStride){
      diagMat[bid*m + tid] = 0;
      if(bid==tid && abs(d_vec[tid])>EPS){
        diagMat[bid*m + tid] = 1/d_vec[tid];
      }
    }
  }
}

float* diagonalizeVector(float* d_vec, int m, bool inverse){
  /*
    return mxm inverted diagonal matrix made of the input vector
  */
  int gridSize = (m*m/BLOCK_SIZE + 1);
  gridSize = min(gridSize, GRID_SIZE);

  int *d_m;
  cudaMalloc(&d_m, sizeof(int));
  cudaMemcpy(d_m, &m, sizeof(int), cudaMemcpyHostToDevice);

  float* diagMatrix;
  cudaMalloc(&diagMatrix, m*m*sizeof(float));

  if (inverse){
    d_GetInverseDiagMatrix<<<gridSize, BLOCK_SIZE>>>(d_vec, diagMatrix, d_m);
  } else {
    d_GetDiagMatrix<<<gridSize, BLOCK_SIZE>>>(d_vec, diagMatrix, d_m);
  }

  cudaFree(&d_m);
  return diagMatrix;
}



float* getPseudoInverse(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH,
  float *d_A, int m, int n){

  float *d_At;
  float alfa = 1.0f;
  float beta = 0;
  cudaMalloc (&d_At , sizeof(float)*m*n);
  cublasStatus_t cublasStat = cublasSgeam(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
    m, n, &alfa, d_A, n, &beta, d_A, m, d_At, m);

  int lwork = 0;
  cusolverStatus_t cusolverStat = cusolverDnSgesvd_bufferSize(cusolverH, m, n, &lwork);

  float *d_work, *d_rwork;
  cudaMalloc((void**)&d_work , sizeof(float)*lwork);

  float *d_S, *d_U, *d_VT;
  int *devInfo;
  cudaMalloc ((void**)&d_S , sizeof(float)*n);
  cudaMalloc ((void**)&d_U , sizeof(float)*m*m);
  cudaMalloc ((void**)&d_VT , sizeof(float)*m*n);
  cudaMalloc ((void**)&devInfo, sizeof(int));

  signed char jobu = 'A'; // all m columns of U
  signed char jobvt = 'A'; // all n columns of VT

  cusolverStat = cusolverDnSgesvd (
    cusolverH,
    jobu,
    jobvt,
    m,
    n,
    d_At,
    m,
    d_S,
    d_U,
    m, // ldu
    d_VT,
    n, // ldvt,
    d_work,
    lwork,
    d_rwork,
    devInfo);
  printf("%d %d %d\n ", cusolverStat, CUSOLVER_STATUS_SUCCESS, CUSOLVER_STATUS_INVALID_VALUE);
  assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);
  cudaFree(&devInfo);
  cudaFree(&d_work);
  cudaFree(&d_rwork);

  /*
    invert each components
    and then multiply to get pseudo-inverse
  */
  float *d_invDiagS = diagonalizeVector(d_S, n, true);
  cudaFree(&d_S);

  float* d_Ainv, *d_Temp;
  cudaMalloc(&d_Temp, n*m*sizeof(float));
  cudaMalloc(&d_Ainv, n*m*sizeof(float));
  matMul(cublasH, d_invDiagS, d_U, d_Temp, n, n, m, true);
  matMul(cublasH, d_VT, d_Temp, d_Ainv, n, n, m, true);
  cudaFree(d_U);
  cudaFree(d_VT);
  cudaFree(d_Temp);

  return d_Ainv;
}
