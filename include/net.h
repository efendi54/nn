#pragma once
#include "layer.h"

// /////////////////
struct FFNet
{
  FFNet(const std::vector<size_t> &p_Arch);
  void Print(void);
  void Feed(const DVec &p_X);
  void BackProp(const DVec &p_Y);
  void Learn(const std::vector<DVec> &p_X,
             const std::vector<DVec> &p_Y,
             const ulong p_Iterations=1000);

  double t_LearnRate;
  double t_Momentum;
  std::vector<Layer> t_Layers;
};

