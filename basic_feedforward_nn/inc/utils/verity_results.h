#ifndef UTILS_VERIFY_RESULTS_H
#define UTILS_VERIFY_RESULTS_H

#include <vector>
#include <cassert>

void verify_result_mul(std::vector<int>& a, std::vector<int>& b, std::vector<int>& c, int N);
void verify_result_add(std::vector<int>& a, std::vector<int>& b, std::vector<int>& c, int N);
void verify_result_scale(std::vector<int>& a, int factor, std::vector<int>& c, int N);

#endif // !UTILS_VERIFY_RESULTS_H
