#include <limits>
#include <vector>
#include <map>

#include "net.h"
#include "mnist_reader.h"

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

  cout << "x-data-size = " << XTrainData.size() << endl;
  cout << "y-data-size = " << YTrainData.size() << endl;
  cout << "x-test-size = " << XTestData.size() << endl;
  cout << "y-test-size = " << YTestData.size() << endl;
  
 CenterData(XTrainData);
  size_t iterations = 20,
         train_size = 50000,
         input_size = XTrainData[0].size(),
         output_size = YTrainData[0].size(),
         hidden_size = 50;

  double learn_rate = 0.3,
         momentum = 0.3;
 
  if(argc>1)
    iterations = atoi(argv[1]);
  if(argc>2)
    learn_rate = atof(argv[2]);
  if(argc>3)
    momentum = atof(argv[3]);
  if(argc>4)
    train_size = atoi(argv[4]);

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

