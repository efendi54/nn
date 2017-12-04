#pragma once
#include "layer.h"
#include <string>

void ReadMnistImages(const std::string &p_FileName,
    std::vector<DVec> &p_X);

void ReadMnistLabels(const std::string &p_FileName,
    std::vector<DVec> &p_Y);

