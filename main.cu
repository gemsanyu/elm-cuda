#include <assert.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "helper_lib.h"
#include "local_lib.h"

using namespace std;

#define IDX2C(i,j,ld) (((i)*(ld))+(j))
#define ROOT 0

int main(int argc, char ** argv){
  MPI_Status status;
  int rank;
  int numOfProcess;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numOfProcess);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  cudaError_t cudaStat; // cudaMalloc cublasStatus
  cublasStatus_t cublasStat; // CUBLAS functions cublasStatus
  cublasHandle_t cublasH; // CUBLAS context
  cusolverDnHandle_t cusolverH; //CUSOLVER context
  cusolverStatus_t cusolverStat;

  cublasStat = cublasCreate (&cublasH); // initialize CUBLAS context
  cusolverStat = cusolverDnCreate(&cusolverH);

  Config config;
  int *rowSplitSize;
  int *rowOffsetSplit;
  int row, rowOffset, col, classNum;
  int hiddenNeuron;
  double alpha;
  string xFileName,yFileName;
  RunningTimeData rt;
  double start, end, realStart, realEnd;

  /*
    cublasgemm param
  */
  float gemmA=1.0f, gemmB=0;

  /*
    Read the config file
    and prepare the row counts for every process
    scatter row
  */
  realStart=MPI_Wtime();
  start = realStart;
  config = readConfig(argv[1]);
  col = config.col;
  classNum = config.classNum;
  hiddenNeuron = config.hiddenNeuron;
  alpha = config.alpha;
  xFileName = config.xFileName;
  yFileName = config.yFileName;
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == ROOT){
    rowSplitSize = create1DArrayInt(numOfProcess);
    rowOffsetSplit = create1DArrayInt(numOfProcess);
    int rowSplitSizeRemainder = config.row % numOfProcess;
    int rowCount = config.row / numOfProcess;
    for (int rankIter=0; rankIter<numOfProcess; rankIter++){
      rowSplitSize[rankIter]=rowCount;
      if (rankIter < rowSplitSizeRemainder){
        rowSplitSize[rankIter]++;
      }
    }

    // Read Offset for the MPI-PO, offset is sum of rows from n=0, n=rank-1
    // Still needs to be multiplied by size of double and number of column
    // col for x, and classNum for y
    rowOffsetSplit[0]=0;
    for(int rankIter=1; rankIter<numOfProcess; rankIter++){
      rowOffsetSplit[rankIter]=rowOffsetSplit[rankIter-1]+rowSplitSize[rankIter-1];
    }
  }

  /*
    Scatter rows and offset information
  */
  MPI_Scatter(rowSplitSize, 1, MPI_INT, &row, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
  MPI_Scatter(rowOffsetSplit, 1, MPI_INT, &rowOffset, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

  printf("ROW %d: %d\n", rank, row);
  printf("ROW OFFSET %d: %d\n", rank, rowOffset);

  /*
    now read the matrix separately by each process.
    try MPI_IO woohoo
    +1 col for bias
  */
  float *X = (float*) malloc(row*(col+1)*sizeof(float));
  MPI_File xFile;
  MPI_File_open(MPI_COMM_WORLD, xFileName.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &xFile);
  MPI_File_read_at(xFile, rowOffset*(col+1)*sizeof(float), X, row*(col+1), MPI_FLOAT, &status);
  MPI_File_close(&xFile);

  float *d_X;
  cudaMalloc(&d_X, row*(col+1)*sizeof(float));
  cublasStat = cublasSetMatrix(row, col+1, sizeof(float), X, row, d_X, row);
  // cudaMemcpy(d_X, X, row*(col+1)*sizeof(float), cudaMemcpyHostToDevice);

  float *Y = (float*) malloc(row*classNum*sizeof(float));
  MPI_File yFile;
  MPI_File_open(MPI_COMM_WORLD, yFileName.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &yFile);
  MPI_File_read_at(yFile, rowOffset*classNum*sizeof(float), Y, row*classNum, MPI_FLOAT, &status);
  MPI_File_close(&yFile);

  float *d_Y;
  cudaMalloc(&d_Y, row*classNum*sizeof(float));
  cublasStat = cublasSetMatrix(row, classNum, sizeof(float), Y, row, d_Y, row);
  // cudaMemcpy(d_Y, Y, row*classNum*sizeof(float), cudaMemcpyHostToDevice);

  end = MPI_Wtime();
  rt.readDataTime = end-start;

  /*
    Generate random weight input and broadcast it
    to all the processes
    remember the row of W_input is col + 1 (1 for bias)
    and the col is the number of hidden neurons
  */
  start = MPI_Wtime();
  float *WInput, *d_WInput;
  if (rank == ROOT){
    WInput = generateWeightInput(col+1, hiddenNeuron);
    cudaMalloc(&d_WInput, (col+1)*hiddenNeuron*sizeof(float));
    cublasStat = cublasSetMatrix(col+1, hiddenNeuron, sizeof(float), WInput, col+1, d_WInput, col+1);
    // cudaMemcpy(d_WInput, WInput, (col+1)*hiddenNeuron*sizeof(float), cudaMemcpyHostToDevice);
  } else {
    cudaMalloc(&d_WInput, (col+1)*hiddenNeuron*sizeof(float));
  }
  MPI_Bcast(d_WInput, (col+1)*hiddenNeuron, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
  end = MPI_Wtime();
  rt.generateWeightTime = end-start;

  /*
    X M*N
    W N*500
    H = X*W = M*500
    A = Ht*H = 500*500

    Start the calculation because life is good
  */
  start = MPI_Wtime();
  float *d_H;
  cudaMalloc(&d_H, row*hiddenNeuron*sizeof(float));
  matMul(cublasH, d_X, d_WInput, d_H, row, col+1, hiddenNeuron, true);
  activationFunction(d_H, row, hiddenNeuron);
  end = MPI_Wtime();
  rt.maxH = end-start;

  start = MPI_Wtime();
	float *d_A;
  cudaMalloc(&d_A, hiddenNeuron*hiddenNeuron*sizeof(float));
  cublasStat = cublasSgemm_v2(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, hiddenNeuron, hiddenNeuron,
    row, &gemmA, d_H, hiddenNeuron, d_H, hiddenNeuron, &gemmB, d_A, hiddenNeuron);
  fillDiag(d_A, hiddenNeuron, hiddenNeuron, 1/config.alpha);
  end = MPI_Wtime();
  rt.maxA = end-start;

  start = MPI_Wtime();
  float *d_Temp;
  cudaMalloc(&d_Temp, hiddenNeuron*classNum*sizeof(float));
  cublasStat = cublasSgemm_v2(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, classNum, hiddenNeuron, row,
    &gemmA, d_Y, classNum, d_H, hiddenNeuron, &gemmB, d_Temp, classNum);
  float *d_Ainv = getPseudoInverse(cublasH, cusolverH, d_A, hiddenNeuron, hiddenNeuron);
  float *d_W;
  cudaMalloc(&d_W, hiddenNeuron*classNum*sizeof(float));
  cublasStat = matMul(cublasH, d_Ainv, d_Temp, d_W, hiddenNeuron, hiddenNeuron, classNum);
  cudaFree(d_Ainv);
  cudaFree(d_Temp);
  end = MPI_Wtime();
  rt.maxW = end-start;

  /*
    Recombining all the output wieghts
    if K=1, then return the W else
      combine the A then combine the W
      we'll try to use gather here, see what we got
  */
  start = MPI_Wtime();
  float *d_WOutput;
  if (numOfProcess == 1){
    d_WOutput = d_W;
  }else{
    // Combining A
    float* d_ACombined;
    if (rank == ROOT){
      cudaMalloc(&d_ACombined, hiddenNeuron*hiddenNeuron*sizeof(float));
    }
    MPI_Reduce(d_A, d_ACombined, hiddenNeuron*hiddenNeuron, MPI_FLOAT, MPI_SUM,
      ROOT, MPI_COMM_WORLD);
    if (rank == ROOT){
      fillDiag(d_ACombined, hiddenNeuron, hiddenNeuron, (numOfProcess-1.0)/config.alpha);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    float *d_TempWOutput;
    if (rank == ROOT){
      cudaMalloc(&d_TempWOutput, hiddenNeuron*classNum*sizeof(float));
    }
    float *d_Temp2;
    cudaMalloc(&d_Temp2, hiddenNeuron*classNum*sizeof(float));
    cublasStat = matMul(cublasH, d_A, d_W, d_Temp2, hiddenNeuron, hiddenNeuron, classNum);
    MPI_Reduce(d_Temp2, d_TempWOutput, hiddenNeuron*classNum, MPI_FLOAT, MPI_SUM, ROOT, MPI_COMM_WORLD);
    cudaFree(d_Temp2);
    if (rank == ROOT){
      cudaMalloc(&d_WOutput, hiddenNeuron*classNum*sizeof(float));
      float *d_AComInv = getPseudoInverse(cublasH, cusolverH, d_ACombined, hiddenNeuron, hiddenNeuron);
      matMul(cublasH, d_AComInv, d_TempWOutput, d_WOutput, hiddenNeuron, hiddenNeuron, classNum);
      cudaFree(d_AComInv);
      cudaFree(d_TempWOutput);
    }
    // temp.resize(0,0);
  }

  /*
    Save the output weight
    and write the output time
  */
  start = MPI_Wtime();
  if (rank == ROOT){
    writeMatrixfToFileBinary(config.wInputFileName, WInput, col+1, hiddenNeuron);

    float *WOutput = (float*) malloc(hiddenNeuron*classNum*sizeof(float));
    cublasStat = cublasGetMatrix(hiddenNeuron, classNum, sizeof(float), d_WOutput, hiddenNeuron, WOutput, hiddenNeuron); // cp d_c - >c
    writeMatrixfToFileBinary(config.wOutputFileName, WOutput, hiddenNeuron, classNum);
  }

  // Summarize the running time
  double readTime,writeTime,genWTime,maxH, maxA, maxW;
  MPI_Reduce(&rt.readDataTime, &readTime, 1, MPI_DOUBLE, MPI_MAX, ROOT, MPI_COMM_WORLD);
  MPI_Reduce(&rt.writeDataTime, &writeTime, 1, MPI_DOUBLE, MPI_MAX, ROOT, MPI_COMM_WORLD);
  MPI_Reduce(&rt.generateWeightTime, &genWTime, 1, MPI_DOUBLE, MPI_MAX, ROOT, MPI_COMM_WORLD);
  MPI_Reduce(&rt.maxH, &maxH, 1, MPI_DOUBLE, MPI_MAX, ROOT, MPI_COMM_WORLD);
  MPI_Reduce(&rt.maxA, &maxA, 1, MPI_DOUBLE, MPI_MAX, ROOT, MPI_COMM_WORLD);
  MPI_Reduce(&rt.maxW, &maxW, 1, MPI_DOUBLE, MPI_MAX, ROOT, MPI_COMM_WORLD);

  realEnd=MPI_Wtime();
  if (rank == ROOT){
    rt.np = numOfProcess;
    rt.row = config.row;
    rt.col = config.col;
    rt.hiddenNeuron = config.hiddenNeuron;
    rt.totalTime = rt.readDataTime + rt.writeDataTime + rt.generateWeightTime+
    rt.maxH + rt.maxA + rt.maxW + rt.combineW;
    rt.realTotalTime = realEnd - realStart;
    printf("Total Time %.5f\n", rt.realTotalTime);
    // writeRunningTimeData(config.runningTimeFileName, rt);
  }

  cudaFree(&d_H);
  cudaFree(&d_X);
  cudaFree(&d_Y);
  cublasStat = cublasDestroy_v2(cublasH);
  cusolverStat = cusolverDnDestroy(cusolverH);
  MPI_Finalize();
}
