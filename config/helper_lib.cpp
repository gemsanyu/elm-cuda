#include<fstream>
#include<iterator>
#include<iostream>
#include<math.h>
#include<sstream>
#include<stdlib.h>
#include<string>
#include<unistd.h>

#include"helper_lib.h"

using namespace std;

Config readConfig(string configFileName){
  ifstream configFile(configFileName);

  Config config;
  string dataName;
  configFile >> dataName;
  config.xFileName = "data/"+dataName+"/training/file_x.bin";
  config.yFileName = "data/"+dataName+"/training/file_y.bin";
  configFile >> config.row >> config.col >> config.classNum >> config.hiddenNeuron;
  configFile >> config.alpha;
  config.wInputFileName = "weight/"+dataName+"/w-in-"+
  to_string(config.hiddenNeuron);
  config.wOutputFileName = "weight/"+dataName+"/w-out-"+
  to_string(config.hiddenNeuron);
  configFile >> config.runningTimeFileName;
  config.runningTimeFileName = "report/"+dataName+"/"+config.runningTimeFileName;

  cout << "Using " << dataName <<"\n";
  cout << "Total Rows : " << config.row <<"; Cols : "<< config.col<<"; ";
  cout << "Class : " << config.classNum<<"\n";
  cout << "Hidden Neuron : " << config.hiddenNeuron <<"\n";
  return config;
}

ConfigTest readConfigTest(string configFileName){
  ifstream configFile(configFileName);

  ConfigTest config;
  string dataName;
  configFile >> dataName;
  config.xFileName = "data/"+dataName+"/test/file_x.bin";
  config.yFileName = "data/"+dataName+"/test/file_y.bin";
  configFile >> config.row >> config.col >> config.classNum >> config.hiddenNeuron;
  configFile >> config.alpha;
  config.wInputFileName = "weight/"+dataName+"/w-in-"+
  to_string(config.hiddenNeuron)+".bin";
  config.wOutputFileName = "weight/"+dataName+"/w-out-"+
  to_string(config.hiddenNeuron)+".bin";
  configFile >> config.accuracyFileName;
  config.accuracyFileName = "report/"+dataName+"/"+config.accuracyFileName;

  cout << "Using " << dataName <<"\n";
  cout << "Total Rows : " << config.row <<"; Cols : "<< config.col<<"; ";
  cout << "Class : " << config.classNum<<"\n";
  cout << "Hidden Neuron : " << config.hiddenNeuron <<"\n";
  cout << "W-input Data: " << config.wInputFileName <<"\n";
  cout << "W-output Data: " << config.wOutputFileName <<"\n";
  cout << "Test X Data: " << config.xFileName <<"\n";
  cout << "Test Y Data: " << config.yFileName <<"\n";
  return config;
}

void writeMatrixToFileText(string fileName, MatrixXdRowMajor X){
  fileName = fileName + ".txt";
  printf("Writing Matrix to file %s\n", fileName);
  printf("Rows = %d, Col = %d\n", X.rows(), X.cols());

  FILE *outFile;
  outFile = fopen(fileName.c_str(), "wb");

  int row = X.rows(), col = X.cols();
  for(int r=0; r< row; r++){
    for(int c=0; c <col; c++){
      fprintf(outFile, "%.5lf", X(r,c));
      if (c==col-1){
        fprintf(outFile, "\n");
      } else {
        fprintf(outFile, " ");
      }
    }
  }
  fclose(outFile);
}

void writeMatrixToFileBinary(string fileName, MatrixXdRowMajor X){
  fileName = fileName + ".bin";
  printf("Writing Matrix to file %s\n", fileName);
  printf("Rows = %d, Col = %d\n", X.rows(), X.cols());

  ofstream outFile(fileName, ios::trunc | ios::binary);
  outFile.write((char *) X.data(), X.rows()*X.cols()*sizeof(double));
  outFile.close();
}

void writeRunningTimeData(string fileName, RunningTimeData rt){
  printf("Writing output data to file %s\n", fileName);
  FILE *outFile;
  if( access( fileName.c_str(), F_OK ) != -1 ) {
    outFile = fopen(fileName.c_str(), "a");
  } else {
    outFile = fopen(fileName.c_str(), "wb");
    fprintf(outFile, "NP,ROW,COL,HIDDEN_NEURON,READ_TIME,WRITE_TIME,GEN_W_TIME,MAX_H,MAX_A,MAX_W,COMBINE_W,TOTAL,REAL_TOTAL\n");
  }
  fprintf(outFile, "%d,%d,%d,%d,%.5lf,%.5lf,%.5lf,%.5lf,%.5lf,%.5lf,%.5lf,%.5lf,%.5lf\n", rt.np,
  rt.row, rt.col, rt.hiddenNeuron, rt.readDataTime,
  rt.writeDataTime, rt.generateWeightTime, rt.maxH,
  rt.maxA, rt.maxW, rt.combineW,  rt.totalTime,
  rt.realTotalTime);

  fclose(outFile);
}

void writeAccuracyData(string fileName, AccuracyData accuracyData){
  printf("Writing accuracy data to file %s\n", fileName);
  FILE *outFile;
  if( access( fileName.c_str(), F_OK ) != -1 ) {
    outFile = fopen(fileName.c_str(), "a");
  } else {
    outFile = fopen(fileName.c_str(), "wb");
    fprintf(outFile, "RMSE,TRUE_ACCURACY\n");
  }
  fprintf(outFile, "%.5lf,%.5lf\n", accuracyData.RMSE, accuracyData.TrueAccuracy);
  fclose(outFile);
}
