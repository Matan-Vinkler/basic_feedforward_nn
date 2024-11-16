#include <iostream>
#include <vector>
#include <algorithm>

#include "inc/Matrix.cuh"
#include "inc/utils/verity_results.h"

int main()
{
    std::cout << "Starting..." << std::endl;
    srand((unsigned int)time(NULL));

    int N = 1 << 10;

    std::vector<float> h_a(N * N);
    std::vector<float> h_b(N * N);
    std::generate(h_a.begin(), h_a.end(), []() { return (float)rand() / (float)(RAND_MAX / 1); });
    std::generate(h_b.begin(), h_b.end(), []() { return (float)rand() / (float)(RAND_MAX / 1); });

    Matrix A(h_a, N);
    Matrix B(h_b, N);
    Matrix C1(N, MatrixInitType::ZERO);
    Matrix C2(N, MatrixInitType::ZERO);
    Matrix C3(N, MatrixInitType::ZERO);

    C1.matrix_add(A, B);
    std::vector<float> result1 = C1.export_to_host();

    int factor = 4;
    C2.matrix_scale(A, factor);
    std::vector<float> result2 = C2.export_to_host();

    C3.matrix_mul(A, B);
    std::vector<float> result3 = C3.export_to_host();

    std::cout << "Calculated CUDA Matrices, now verifying..." << std::endl;

    verify_result_add(h_a, h_b, result1, N);
    verify_result_scale(h_a, factor, result2, N);
    verify_result_mul(h_a, h_b, result3, N);
    
    std::cout << "COMPLETED SUCCESSFULLY" << std::endl;

    return 0;
}