#include "tools.h"
#include <sstream>
#include <fstream>
#include <map>
#include <algorithm>

using namespace std;

// /////////////////
vector<string> Split(const string &s, 
                     const char delim) 
{
  stringstream ss(s);
  string item;
  vector<string> tokens;
  while (getline(ss, item, delim)) 
    tokens.push_back(item);
  return tokens;
}

// /////////////////
void Split(const string &s, 
    const char* delim, 
    vector<string> & v)
{
  char * dup = strdup(s.c_str());
  char * token = strtok(dup, delim);
  while(token != NULL){
    v.push_back(string(token));
    token = strtok(NULL, delim);
  }
  free(dup);
}

// /////////////////
void ReadLetterData(const string &p_FileName,
    std::vector<DVec> &p_X,
    std::vector<DVec> &p_Y)
{
  ifstream file(p_FileName);
  string line;
  stringstream ss;

  std::map<char, DVec> clsMap;
  const string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  for(size_t i=0; i<alphabet.length(); ++i)
  {
    clsMap[alphabet[i]] = DVec(26);
    clsMap[alphabet[i]][i] = 1;
  }

  while( getline(file, line) )
  {
    vector<string> xyToken = Split(line, ',');
    p_Y.push_back(clsMap[xyToken[0][0]]);
    DVec x;
    for(size_t i=1; i<xyToken.size(); ++i)
      x.push_back(std::stod(xyToken[i]));
    p_X.push_back(x);
  }
}

// /////////////////
void ReadData(const string &p_FileName,
    std::vector<DVec> &p_X,
    std::vector<DVec> &p_Y)
{
  ifstream file(p_FileName);
  string line;
  stringstream ss;
  ulong lcnt=0;
  std::vector<DVec> *pvecs[2] = {&p_X, &p_Y};
  while( getline(file, line) )
  {
    vector<string> xyToken = Split(line, ' ');
    DVec dvec;
    for(size_t i=0; i<xyToken.size(); ++i)
    {
      try
      {
        double x = std::stod(xyToken[i]);
        dvec.push_back(x);
      }
      catch(std::exception &e)
      {
        p_X.clear();
        p_Y.clear();
        cerr << "Exception (" << e.what() << "), in data-file at line:" 
             << lcnt << " for token :" << xyToken[i] << endl;
        throw e;
      }
    }
    if(dvec.size())
      pvecs[lcnt%2]->push_back(dvec);
    ++lcnt;
  }

  if(p_X.size() != p_Y.size())
  {
    std::runtime_error e("Amount of x-data items do not match y-data items !!");
    throw e;
  }
}


// /////////////////
template <class T>
void EndianSwap(T *objp)
{
  unsigned char *memp = reinterpret_cast<unsigned char*>(objp);
  std::reverse(memp, memp + sizeof(T));
}

// /////////////////
void ReadMnistImages(const string &p_FileName,
                     std::vector<DVec> &p_X)
{
  ifstream file(p_FileName, std::ios::binary);
  if(!file.is_open())
    return;

  int magic, items, rows, cols;
  file.read( reinterpret_cast<char*>(&magic) , sizeof(int) );
  file.read( reinterpret_cast<char*>(&items) , sizeof(int) );
  file.read( reinterpret_cast<char*>(&rows) , sizeof(int) );
  file.read( reinterpret_cast<char*>(&cols) , sizeof(int) );
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
//    PrintDVec(x);
    p_X.push_back(x);
  }
  assert(--cnt==items);
}

// /////////////////
void ReadMnistLabels(const string &p_FileName,
                     std::vector<DVec> &p_Y)
{
  ifstream file(p_FileName, std::ios::binary);
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
    //PrintDVec(y);
    p_Y.push_back(y);
  }
}

