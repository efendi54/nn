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

  DVec t_In;        ///< all neurons inputs
  DVec t_Out;       ///< all neurons outputs
  DVec t_DOut;      ///< all neurons outputs passed in through derivate out-function (see also t_Func)
  DVec t_Err;       ///< all neurons errors
  DVec t_Bias;      ///< all neurons biases
  DVec t_BiasDelta; ///< deltas for all biases for recalculation of deltas during learning 
  Func t_Func;      ///< function to be used for calculating neuron outputs
  Func t_DFunc;     ///< derivate function of t_Func to be used for calculating derivate-outputs

  std::vector<DVec> t_Weights;      ///< weighted connections from this layer to previous layer (in direction input to output)
  std::vector<DVec> t_WeightDeltas; ///< deltas for recalculation of new weights during learning
};

