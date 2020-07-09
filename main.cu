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
  cudaMemcpy(d_X, X, row*(col+1)*sizeof(float), cudaMemcpyHostToDevice);

  float *Y = (float*) malloc(row*classNum*sizeof(float));
  MPI_File yFile;
  MPI_File_open(MPI_COMM_WORLD, yFileName.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &yFile);
  MPI_File_read_at(yFile, rowOffset*classNum*sizeof(float), Y, row*classNum, MPI_FLOAT, &status);
  MPI_File_close(&yFile);

  float *d_Y;
  cudaMalloc(&d_Y, row*classNum*sizeof(float));
  cudaMemcpy(d_Y, Y, row*classNum*sizeof(float), cudaMemcpyHostToDevice);

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
    cudaMemcpy(d_WInput, WInput, (col+1)*hiddenNeuron*sizeof(float), cudaMemcpyHostToDevice);
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

  float *d_H;
  cudaMalloc(&d_H, row*hiddenNeuron*sizeof(float));
  matMul(cublasH, d_X, d_WInput, d_H, row, col+1, hiddenNeuron, true);
  activationFunction(d_H, row, hiddenNeuron);

  MPI_Barrier();

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
  // cublasStat = cublasDestroy_v2(cublasH);
  // cusolverStat = cusolverDnDestroy(cusolverH);
  MPI_Finalize();
}
