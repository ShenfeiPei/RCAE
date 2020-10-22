#ifndef _fun_H
#define _fun_H

#include <iostream>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <cmath>

// std::vector<double> EProjSimplex_new_cpp(std::vector<double> &v, int k);
void EProjSimplex_new_cpp(std::vector<double> &v, int k, std::vector<double> &x);
std::vector<std::vector<double>> EProjSimplex_new_M(std::vector<std::vector<double>> &V, int k);

#endif
