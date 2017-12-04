#pragma once
#include <vector>
#include <string>

#include "util.h"
#include "data_reader.h"
 
typedef std::vector<DVec> Data;
typedef std::pair<Data, Data> XYData;
typedef std::pair<std::string, std::string> FileNames;

class MNistReader : public DataReader<FileNames, XYData>
{
  public:
    MNistReader();
    ~MNistReader();

    void Read(const FileNames &p_FileNames,
              XYData &p_XYData);

  private:
    void ReadImages(const std::string &p_FileName, Data &p_X);
    void ReadLabels(const std::string &p_FileName, Data &p_Y);
};
