#ifndef HLP_LIB_H
#define HLP_LIB_H

#include<string>

struct Config{
  std::string xFileName, yFileName;
  int row, col, classNum, hiddenNeuron;
  double alpha;
  std::string wInputFileName, wOutputFileName, runningTimeFileName;
};

struct ConfigTest{
  std::string xFileName, yFileName;
  int row, col, classNum, hiddenNeuron;
  double alpha;
  std::string wInputFileName, wOutputFileName, accuracyFileName;
};


struct RunningTimeData{
  int np, row, col, hiddenNeuron;
  double readDataTime, writeDataTime, generateWeightTime, maxH, maxA, maxW, combineW, totalTime, realTotalTime;
};

struct AccuracyData{
  double RMSE, TrueAccuracy;
};

char* create1DArrayChar(int sizeX);
int* create1DArrayInt(int sizeX);
float* generateWeightInput(int row, int col);
Config readConfig(std::string configFileName);
ConfigTest readConfigTest(std::string configFileName);
void writeRunningTimeData(std::string fileName, RunningTimeData rt);
void writeAccuracyData(std::string fileName, AccuracyData accuracyData);
#endif
