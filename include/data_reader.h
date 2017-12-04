#pragma once

#include <string>

template <class WHERE, class WHAT>
class DataReader
{
  public:
    DataReader(){}
    virtual ~DataReader(){};

    virtual void Read(const WHERE &where,
                      WHAT &data) = 0;
};

