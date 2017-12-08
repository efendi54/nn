#pragma once
#include <vector>
#include <string>

#include "util/util.h"
#include "data/data_reader.h"
 
typedef std::vector<DVec> Data;
typedef std::pair<Data, Data> XYData;
typedef std::pair<std::string, std::string> FileNames;

/**
 * \brief class to read in data from mnist data-source as described in
 *        http://yann.lecun.com/exdb/mnist/
 * */
class MNistReader : public DataReader<FileNames, XYData>
{
  public:
    MNistReader();
    ~MNistReader();

    /**
     * \brief Reads in train and test data from provided sources (files)
     * \param p_FileNames pair of file-names (train-/test-data)
     * \param p_XYData target where to put in read data (train-/test-data)
     * */
    void Read(const FileNames &p_FileNames,
              XYData &p_XYData);

  private:
    void ReadImages(const std::string &p_FileName, Data &p_X);
    void ReadLabels(const std::string &p_FileName, Data &p_Y);
};
