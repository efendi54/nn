#pragma once
#include "layer.h"
#include <string>

void ReadData(const std::string &p_FileName,
              std::vector<DVec> &p_X,
              std::vector<DVec> &p_Y);

void ReadLetterData(const std::string &p_FileName,
    std::vector<DVec> &p_X,
    std::vector<DVec> &p_Y);

void ReadMnistImages(const std::string &p_FileName,
    std::vector<DVec> &p_X);

void ReadMnistLabels(const std::string &p_FileName,
    std::vector<DVec> &p_Y);

