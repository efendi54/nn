#include <cmath>

#include "net.h"
#include <stdio.h>

using namespace std;


FFNet::FFNet(const std::vector<size_t> &p_Arch) 
{
  t_LearnRate = 0.3;
  t_Momentum = 0.3;

  for(size_t a=0; a<p_Arch.size(); ++a)
  {
    size_t con = 0;
    if( a > 0 )
      con = t_Layers.back().t_In.size();
    t_Layers.push_back(Layer(p_Arch[a], con, Sigmoid, DSigmoid));
  }
}

void FFNet::Print(void)
{
  for(size_t l=0; l<t_Layers.size(); ++l)
  {
    cout << "----------" << endl;
    t_Layers[l].Print();
  }
}

void FFNet::Feed(const DVec &p_X)
{
  t_Layers[0].t_Out = p_X;
  for(ushort li=1; li<t_Layers.size(); ++li)
    t_Layers[li].Feed(t_Layers[li-1].t_Out);
}

void FFNet::BackProp(const DVec &p_Y)
{
  // calc errors first
  for(ushort li=t_Layers.size()-1; li>0; --li)
  {
    const bool isOutLayer = (li == (t_Layers.size()-1));
    for(ushort ni=0; ni<t_Layers[li].t_Out.size(); ++ni)
    {
      if(isOutLayer)
      {
        t_Layers[li].t_NetErr[ni] = abs((p_Y[ni] - t_Layers[li].t_Out[ni]));
        t_Layers[li].t_Err[ni] = (p_Y[ni] - t_Layers[li].t_Out[ni]) * t_Layers[li].t_DOut[ni];
      }
      else
      {
        double esum = 0;
        for(ushort pi=0; pi<t_Layers[li+1].t_Out.size(); ++pi)
          esum += t_Layers[li+1].t_Weights[pi][ni] * t_Layers[li+1].t_Err[pi];
        t_Layers[li].t_Err[ni] = esum * t_Layers[li].t_DOut[ni];
      }
    }
  }

  for(ushort li=1; li<t_Layers.size(); ++li)
  {
    for(ushort ni=0; ni<t_Layers[li].t_Out.size(); ++ni)
    {
      for(ushort wi=0; wi<t_Layers[li].t_Weights[ni].size(); ++wi)
      {
        const double wdelta = t_LearnRate * t_Layers[li].t_Err[ni] * t_Layers[li-1].t_Out[wi];
        t_Layers[li].t_Weights[ni][wi] += wdelta + (t_Momentum * t_Layers[li].t_WeightDeltas[ni][wi]);
        t_Layers[li].t_WeightDeltas[ni][wi] = wdelta;
      }

      const double biasDelta = t_LearnRate * t_Layers[li].t_Err[ni];
      t_Layers[li].t_Bias[ni] +=  biasDelta + (t_Momentum * t_Layers[li].t_BiasDelta[ni]);
      t_Layers[li].t_BiasDelta[ni] = biasDelta;
    }
  }
}

void FFNet::ShowSummedNetOutErr(void)
{
  if (t_Layers.empty())
    return;
  double nerr = 0;
  for(size_t i=0; i<t_Layers.back().t_NetErr.size(); ++i)
    nerr += t_Layers.back().t_NetErr[i];
  cout << endl << "summed output-layer-error = " << nerr << endl;
}

void FFNet::Learn(const vector<DVec> &p_X,
                  const vector<DVec> &p_Y,
                  const ulong p_Iterations)
{
  cout << "learning data-size=" << p_X.size() << " iterations= " << p_Iterations << endl;
  for(ulong c=0; c<p_Iterations; ++c)
  {
    for(size_t di=0; di<p_X.size(); ++di)
    {
      Feed(p_X[di]);
      BackProp(p_Y[di]);
      printf("\rxi=%d", (int)di+1);
      fflush(stdout);
    }
    ShowSummedNetOutErr();
  }
}

