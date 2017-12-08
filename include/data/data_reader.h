#pragma once

/**
 * \brief Base-class for implementing any data reader (e.g. to read in 
 *        training data which is provided in some specific data-source).
 *        This class should be used to overwrite its Read(..) method 
 *        to implement own data-readers by passing in appropriate 
 *        data-structures for WHERE and WHAT
 * */
template <class WHERE, class WHAT>
class DataReader
{
  public:
    DataReader(){}
    virtual ~DataReader(){};

    virtual void Read(const WHERE &where,
                      WHAT &data) = 0;
};

