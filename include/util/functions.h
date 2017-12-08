#pragma once

#include "util/util.h"

typedef double (*Func) (const double &);

double Identity(const double &p_X);
double DIdentity(const double &p_X);
double Tanh(const double &p_X);
double DTanh(const double &p_X);
double Sigmoid(const double &p_X);
double DSigmoid(const double &p_X);
double SmallRandomNumber(const double p_Min=0.1);
void PrintDVec(const std::vector<double> &p_Vec, const bool p_Newline=true);

