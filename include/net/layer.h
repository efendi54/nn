#pragma once

#include "util/functions.h"
#include "util/util.h"

/**
  \brief Structure to hold data of neuron-layer for layer oriented nets
         (feed-forward-nets)
*/
struct Layer
{
  Layer(const size_t p_Size=1, 
        const size_t p_Connections=0,
        Func p_Func=Sigmoid,
        Func p_DFunc=DSigmoid);
 
  void Connect(const size_t &p_Connections);
  void Feed(const DVec &p_X);
  void Print(void);

  DVec t_In;
  DVec t_Out;
  DVec t_DOut;
  DVec t_Err;
  DVec t_Bias;
  DVec t_BiasDelta;

  Func t_Func;
  Func t_DFunc;

  std::vector<DVec> t_Weights;
  std::vector<DVec> t_WeightDeltas;
};

