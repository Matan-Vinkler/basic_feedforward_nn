#ifndef UTILS_VERIFY_RESULTS_H
#define UTILS_VERIFY_RESULTS_H

#include <vector>
#include <cassert>

void verify_result_mul(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c, int NUM_ROWS, int NUM_COMMON, int NUM_COLS);
void verify_result_add(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c, int NUM_ROWS, int NUM_COLS);
void verify_result_scale(std::vector<float>& a, int scalar, std::vector<float>& c, int NUM_ROWS, int NUM_COLS);

#endif // !UTILS_VERIFY_RESULTS_H
