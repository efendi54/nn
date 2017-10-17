#include "functions.h"

using namespace std;

// /////////////////
double Identity(const double &p_X)
{
  return p_X;
}

// /////////////////
double DIdentity(const double &p_X)
{
  return 1;
}

// /////////////////
double Tanh(const double &p_X)
{
  return tanh(p_X);
}

// /////////////////
double DTanh(const double &p_X)
{
  double y = tanh(p_X);
  return (1 - (y * y));
}

// /////////////////
double Sigmoid(const double &p_X)
{
  return (double)(1 / (1 + exp(-p_X)));
}

// /////////////////
double DSigmoid(const double &p_X)
{
  return Sigmoid(p_X) * (1-Sigmoid(p_X)); 
}

// /////////////////
double SmallRandomNumber(const double p_Min)
{
  const ulong max_range = RAND_MAX;
  double rnd = (double)(rand() % max_range) / max_range;
  return p_Min * rnd;
}

// /////////////////
void PrintDVec(const vector<double> &p_Vec,
               const bool p_Newline)
{
  for(ushort n=0; n<p_Vec.size(); ++n)
    cout << p_Vec[n] << " ";
  if(p_Newline)
    cout << endl;
}
