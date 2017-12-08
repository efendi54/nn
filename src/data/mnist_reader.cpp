#include <fstream>
#include <algorithm>
#include <assert.h>
#include <iostream>

#include "data/mnist_reader.h"

using namespace std;

// /////////////////
template <class T>
void EndianSwap(T *objp)
{
  unsigned char *memp = reinterpret_cast<unsigned char*>(objp);
  std::reverse(memp, memp + sizeof(T));
}

// /////////////////
MNistReader::MNistReader()
{
}

// /////////////////
MNistReader::~MNistReader()
{
}

// /////////////////
void MNistReader::ReadImages(const std::string &p_FileName,
                             Data &p_X)
{
  ifstream file(p_FileName.c_str(), std::ios::binary);
  if(!file.is_open())
    return;

  int magic, items, rows, cols;
  file.read( reinterpret_cast<char*>(&magic), sizeof(int) );
  file.read( reinterpret_cast<char*>(&items), sizeof(int) );
  file.read( reinterpret_cast<char*>(&rows), sizeof(int) );
  file.read( reinterpret_cast<char*>(&cols), sizeof(int) );
  EndianSwap(&magic);
  EndianSwap(&items);
  EndianSwap(&rows);
  EndianSwap(&cols);

  unsigned long cnt = 0;
  char data;
  while(!file.eof() && ++cnt<=items)
  {
    DVec x(rows*cols);
    for(int d=0; d<(rows*cols); ++d)
    {
      file.read(&data, 1);
      x[d] = (unsigned char)data;
    }
    p_X.push_back(x);
  }
  assert(--cnt==items);
}

// /////////////////
void MNistReader::ReadLabels(const std::string &p_FileName,
                             Data &p_Y)
{
  ifstream file(p_FileName.c_str(), std::ios::binary);
  if(!file.is_open())
    return;

  int magic,
      items;
  file.read( reinterpret_cast<char*>(&magic) , sizeof(int) );
  file.read( reinterpret_cast<char*>(&items) , sizeof(int) );
  EndianSwap(&magic);
  EndianSwap(&items);

  unsigned long cnt = 0;
  char data;
  while(!file.eof() && (++cnt<=items))
  {
    file.read(&data, 1);
    assert(data<10);
    DVec y(10); y[data] = 1;
    p_Y.push_back(y);
  }
}

// /////////////////
void MNistReader::Read(const FileNames &p_FileNames,
                       XYData &p_Data)
{
  cout << "MNist-Reader reading data:" << endl;
  cout << p_FileNames.first << endl;
  cout << p_FileNames.second << ".." << endl;
  ReadImages(p_FileNames.first, p_Data.first);
  ReadLabels(p_FileNames.second, p_Data.second);
  cout << "data-size = " << p_Data.first.size() << endl;
  cout << "label-size = " << p_Data.second.size() << endl;
}

