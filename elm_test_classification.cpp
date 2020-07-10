#include<fstream>
#include<iomanip>
#include<iostream>
#include<math.h>
#include<mpi.h>
#include<string>
#include"helper_lib.h"
#include"Eigen/Dense"

using namespace std;
using Eigen::MatrixXf;

#define ROOT 0

float sqr(float x){
  return x*x;
}

int main(int argc, char ** argv){
  MPI_Status status;
  int rank;
  int numOfProcess;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numOfProcess);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int *rowSplitSize;
  int *rowOffsetSplit;
  int totalRow, row, rowOffset, col, classNum;
  int hiddenNeuron;
  double alpha;
  string xFileName,yFileName;

  ConfigTest config = readConfigTest(argv[1]);
  col = config.col;
  classNum = config.classNum;
  hiddenNeuron = config.hiddenNeuron;
  alpha = config.alpha;
  xFileName = config.xFileName;
  yFileName = config.yFileName;

  if (rank == ROOT){
    totalRow = config.row;
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


  /*
  Read data file
  As in training, 1 extra coloumn of Ones will be added for X
  for the bias in W-input rows
  */
  MatrixXfRowMajor testX = MatrixXf(row, col);
  MPI_File xFile;
  MPI_File_open(MPI_COMM_WORLD, xFileName.c_str(), MPI_MODE_RDONLY,
  MPI_INFO_NULL, &xFile);
  MPI_File_read_at(xFile, rowOffset*col*sizeof(float), testX.data(), row*col,
  MPI_FLOAT, &status);
  testX.conservativeResize(Eigen::NoChange, col+1);
  testX.col(col) = Eigen::VectorXf::Ones(row);
  MPI_File_close(&xFile);

  MatrixXfRowMajor testY = MatrixXf(row,classNum);
  MPI_File yFile;
  MPI_File_open(MPI_COMM_WORLD, yFileName.c_str(), MPI_MODE_RDONLY,
  MPI_INFO_NULL, &yFile);
  MPI_File_read_at(yFile, rowOffset*classNum*sizeof(float), testY.data(), row*classNum,
  MPI_FLOAT, &status);
  MPI_File_close(&yFile);

  // Make testYVec, to get the actual class
  // for easier comparison with the predicted class
  // for TrueAccuracy
  Eigen::VectorXf testYVec = Eigen::VectorXf(row);
  Eigen::VectorXf::Index maxIdx;
  for (int r=0;r<row;r++){
    testY.row(r).maxCoeff(&maxIdx);
    testYVec(r)=maxIdx;
  }

  /*
    Read Weight
  */
  MatrixXfRowMajor wInput = MatrixXf(col+1, hiddenNeuron);
  MPI_File wInputFile;
  MPI_File_open(MPI_COMM_WORLD, config.wInputFileName.c_str(), MPI_MODE_RDONLY,
  MPI_INFO_NULL, &wInputFile);
  MPI_File_read(wInputFile, wInput.data(), (col+1)*hiddenNeuron, MPI_FLOAT,
  &status);
  MPI_File_close(&wInputFile);

  MatrixXfRowMajor wOutput = MatrixXf(hiddenNeuron, classNum);
  MPI_File wOutputFile;
  MPI_File_open(MPI_COMM_WORLD, config.wOutputFileName.c_str(), MPI_MODE_RDONLY,
  MPI_INFO_NULL, &wOutputFile);
  MPI_File_read(wOutputFile, wOutput.data(), hiddenNeuron*classNum, MPI_FLOAT,
  &status);
  MPI_File_close(&wOutputFile);

  /*
    Feed Forward to get prediction in matrix form
    and in vector form (containing exact class prediction)
  */
  MatrixXfRowMajor testH;
  testH.noalias() = testX * wInput;
	testH = testH.unaryExpr(&activationFunction);

  MatrixXfRowMajor predY;
  predY.noalias() = testH * wOutput;
  Eigen::VectorXf predYVec = Eigen::VectorXf(row);
  for (int r=0;r<row;r++){
    predY.row(r).maxCoeff(&maxIdx);
    predYVec(r)=maxIdx;
  }
  // for (int r=0;r<=10;r++){
  //   cout << predYVec.row(r)<<"\n";
  // }


  /*
    Calculating Accuracy RMSE and True Accuracy
  */
  double localRMSE=0, totalRMSE=0;
  AccuracyData accuracyData;
  testY -= predY;
  testY.unaryExpr(&sqr);
  for (int r=0; r<row; r++){
    localRMSE += testY.row(r).sum()/(double)classNum;
  }

  MPI_Reduce(&localRMSE, &totalRMSE, 1, MPI_DOUBLE, MPI_SUM, ROOT,
    MPI_COMM_WORLD);
  if (rank==ROOT){
    accuracyData.RMSE = sqrt(totalRMSE/ (double)totalRow);
  }

  int localCount=0, totalCount=0;
  for (int r=0; r<row; r++){
    if (testYVec.row(r) == predYVec.row(r)){
      localCount++;
    }
  }
  MPI_Reduce(&localCount, &totalCount, 1, MPI_INT, MPI_SUM, ROOT,
    MPI_COMM_WORLD);
  if (rank == ROOT){
      accuracyData.TrueAccuracy= (double)totalCount/(double)totalRow;
      cout << "RMSE: " << accuracyData.RMSE <<" \n";
      cout << "True Acc: " << accuracyData.TrueAccuracy <<" \n";
      writeAccuracyData(config.accuracyFileName, accuracyData);
  }

  MPI_Finalize();
}
