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

int* create1DArrayInt(int sizeX){
    int*  array =(int*) malloc(sizeX * sizeof(int));
    return array;
}

char* create1DArrayChar(int sizeX){
    char*  array =(char*) malloc(sizeX * sizeof(char));
    return array;
}

float* generateWeightInput(int row, int col){
	srand(time(NULL));
	float *weightInput = (float*) malloc(row*col*sizeof(float));
	for(int i=0;i<row*col;i++){
		weightInput[i]= static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	}
	return weightInput;
}

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
