#include <limits>
#include <vector>
#include <map>
#include <algorithm>

#include <math.h>
#include <assert.h>
//#include <vector>
#include <iostream>


#include "net/net.h"
#include "data/mnist_reader.h"

using namespace std;

ushort fmax(const DVec &p_Vec)
{
  return std::distance(p_Vec.begin(), 
      std::max_element(p_Vec.begin(), p_Vec.end()));
}

// /////////////////
void CenterData(vector<DVec> &p_XData)
{
  // center data for each feature:
  // find mean and subtract mean from data
  for(size_t i=0; i<p_XData.size(); ++i)
  {
    double mean = 0;
    for(size_t f=0; f<p_XData[i].size(); ++f)
      mean += p_XData[i][f];
    mean /= p_XData[i].size();
    for(size_t f=0; f<p_XData[i].size(); ++f)
      p_XData[i][f] -= mean;
  }
}

// /////////////////
void ScaleData(vector<DVec> &p_XData)
{
  const size_t dataSize = p_XData.size();
  assert(dataSize>0);
  // scale data for each feature (into range [0,1]):
  DVec maxVec(p_XData[0].size());
  for(size_t i=0; i<p_XData.size(); ++i)
    for(size_t f=0; f<p_XData[i].size(); ++f)
      if(p_XData[i][f] > maxVec[f])
        maxVec[f] = p_XData[i][f];

  for(size_t i=0; i<p_XData.size(); ++i)
    for(size_t f=0; f<p_XData[i].size(); ++f)
      if(maxVec[f] > 0)
        p_XData[i][f] /= maxVec[f];
}

// /////////////////
void NormalizeMean(vector<DVec> &p_XData)
{
  DVec meanVec(p_XData[0].size(), 0);
  DVec maxVec(p_XData[0].size(), std::numeric_limits<short>::min());
  DVec minVec(p_XData[0].size(), std::numeric_limits<short>::max());
  for(size_t i=0; i<p_XData.size(); ++i)
  {
    for(size_t f=0; f<p_XData[i].size(); ++f)
    {
      meanVec[f] += p_XData[i][f];
      if(p_XData[i][f] > maxVec[f])
        maxVec[f] = p_XData[i][f];
      if(p_XData[i][f] < minVec[f])
        minVec[f] = p_XData[i][f];
    }
  }

  for(size_t f=0; f<meanVec.size(); ++f)
    meanVec[f] /= p_XData.size();

  for(size_t i=0; i<p_XData.size(); ++i)
  {
    for(size_t f=0; f<p_XData[i].size(); ++f)
    {
      double mean = meanVec[f];
      double maxDiff = maxVec[f] - minVec[f];
      p_XData[i][f] = (p_XData[i][f] - mean) / ((maxDiff!=0)?maxDiff:1);
    }
  }
}

// /////////////////
void Standardize(vector<DVec> &p_XData)
{
  DVec meanVec(p_XData[0].size(), 0);
  for(size_t i=0; i<p_XData.size(); ++i)
    for(size_t f=0; f<p_XData[i].size(); ++f)
      meanVec[f] += p_XData[i][f];
  for(size_t f=0; f<meanVec.size(); ++f)
    meanVec[f] /= p_XData.size();

  DVec varianceVec(p_XData[0].size(), 0);
  for(size_t i=0; i<p_XData.size(); ++i)
    for(size_t f=0; f<p_XData[i].size(); ++f)
      varianceVec[f] += pow(p_XData[i][f] - meanVec[f], 2);
  for(size_t f=0; f<varianceVec.size(); ++f)
  {
    varianceVec[f] /= p_XData.size()-1;
    varianceVec[f] = sqrt(varianceVec[f]);
  }

  for(size_t i=0; i<p_XData.size(); ++i)
    for(size_t f=0; f<p_XData[i].size(); ++f)
    {
      if ((meanVec[f]!=0) && (varianceVec[f]!=0))
        p_XData[i][f] = (p_XData[i][f] - meanVec[f]) / varianceVec[f];
    }
}

// /////////////////
int main(int argc, char *argv[])
{
  srand(time(NULL));

  vector<DVec> XTrainData, XTestData;
  vector<DVec> YTrainData, YTestData;

  FileNames trainDataFile; 
  FileNames testDataFile;
  XYData trainData; 
  XYData testData; 

  trainDataFile.first = "datasets/minst/train/train-images-idx3-ubyte";
  trainDataFile.second = "datasets/minst/train/train-labels-idx1-ubyte";
  testDataFile.first = "datasets/minst/test/t10k-images-idx3-ubyte";
  testDataFile.second = "datasets/minst/test/t10k-labels-idx1-ubyte";

  try
  {
    MNistReader reader;
    reader.Read(trainDataFile, trainData);
    reader.Read(testDataFile, testData);
    XTrainData = trainData.first;
    YTrainData = trainData.second;
    XTestData = testData.first;
    YTestData = testData.second;
  }
  catch(std::exception &e)
  {
    cerr << "error during data-parsing:" << e.what() << endl;
    return 1;
  }

  size_t iterations = 1,
         train_size = 60000,
         input_size = XTrainData[0].size(),
         output_size = YTrainData[0].size(),
         hidden_size = 50;

  double learn_rate = 0.3,
         momentum = 0.3;

  if(argc>1)
  { 
    if (atoi(argv[1]) & 0x1)
    {
      NormalizeMean(XTrainData);
      NormalizeMean(XTestData);
    }
    if (atoi(argv[1]) & 0x2)
    {
      Standardize(XTrainData);
      Standardize(XTestData);
    }
    if (atoi(argv[1]) & 0x4)
    {
      ScaleData(XTrainData);
      ScaleData(XTestData);
    }
    if (atoi(argv[1]) & 0x8)
    {
      CenterData(XTrainData);
      CenterData(XTestData);
    }
  }
  if(argc>2) learn_rate = atof(argv[2]);
  if(argc>3) momentum = atof(argv[3]);
  if(argc>4) train_size = atoi(argv[4]);

  assert(XTrainData.size() == YTrainData.size());
  assert(XTrainData.size()>0 && train_size<=XTrainData.size());
  assert(XTestData.size()>0);
  assert(YTestData.size()>0);

  XTrainData = vector<DVec>(XTrainData.begin(), XTrainData.begin() + train_size);
  YTrainData = vector<DVec>(YTrainData.begin(), YTrainData.begin() + train_size);

  vector<size_t> arch;
  arch.push_back(input_size);
  arch.push_back(hidden_size);
  arch.push_back(output_size);
  FFNet net(arch);
  net.t_LearnRate = learn_rate;
  net.t_Momentum = momentum;
  net.Learn(XTrainData,YTrainData,iterations);

  ulong correct = 0;
  for(size_t di=0; di<XTestData.size(); ++di)
  {
    net.Feed(XTestData[di]);
    double should = fmax(YTestData[di]);
    double is = fmax(net.t_Layers.back().t_Out);
    correct += (should == is);
    //    cout << should << "<->" << is << endl;
  }
  double acc = (double)(correct)/XTestData.size();
  cout << " accuracy=" << acc << endl;

  //  net.Print();
  return 0;
}

