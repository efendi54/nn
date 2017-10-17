#include "layer.h"

using namespace std;

Layer::Layer(const size_t p_Size, 
             const size_t p_Connections,
             Func p_Func,
             Func p_DFunc)
{
  DVec tmp;
  for(ushort n=0; n<p_Size; ++n)
  {
    t_In.push_back(0);
    t_Out.push_back(0);
    t_DOut.push_back(0);
    t_Err.push_back(0); 
    t_Bias.push_back(0);
    t_BiasDelta.push_back(0);
  }
  if( p_Connections > 0 )
    Connect(p_Connections);
  t_Func = p_Func;
  t_DFunc = p_DFunc;
}

void Layer::Connect(const size_t &p_Connections)
{
  for(ushort n=0; n<t_In.size(); ++n)
  {
    t_Weights.push_back(DVec());
    t_WeightDeltas.push_back(DVec());
    for(ushort w=0; w<p_Connections; ++w)
    {
      t_Weights[n].push_back(SmallRandomNumber());
      t_WeightDeltas[n].push_back(0);
    }
  }
}

void Layer::Feed(const DVec &p_X)
{
  for(ushort ni=0; ni<t_In.size(); ++ni)
  {
    t_In[ni] = 0;
    const size_t ws = t_Weights[ni].size();
    const size_t xs = p_X.size();
    assert(ws==xs);
    for(ushort wi=0; wi<t_Weights[ni].size(); ++wi)
      t_In[ni] += t_Weights[ni][wi] * p_X[wi];
    t_In[ni] += t_Bias[ni];
    t_Out[ni] = t_Func(t_In[ni]);
    t_DOut[ni] = t_DFunc(t_In[ni]);
  }
}

void Layer::Print(void)
{
  cout << "I:"; PrintDVec(t_In);
  cout << "O:"; PrintDVec(t_Out);
  cout << "E:"; PrintDVec(t_Err);
  cout << "B:"; PrintDVec(t_Bias);
  cout << endl;
  for(ushort w=0; w<t_Weights.size(); ++w)
    PrintDVec(t_Weights[w]);
}

