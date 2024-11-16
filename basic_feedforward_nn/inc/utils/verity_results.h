#ifndef UTILS_VERIFY_RESULTS_H
#define UTILS_VERIFY_RESULTS_H

#include <vector>
#include <cassert>

void verify_result_mul(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c, int N);
void verify_result_add(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c, int N);
void verify_result_scale(std::vector<float>& a, int factor, std::vector<float>& c, int N);

#endif // !UTILS_VERIFY_RESULTS_H
