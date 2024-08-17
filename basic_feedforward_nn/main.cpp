#include <iostream>
#include <vector>
#include <algorithm>

#include "inc/Matrix.cuh"
#include "inc/utils/verity_results.h"

int main()
{
    std::cout << "Starting..." << std::endl;

    int N = 1 << 10;

    std::vector<int> h_a(N * N);
    std::vector<int> h_b(N * N);
    std::generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
    std::generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

    Matrix A(h_a, N);
    Matrix B(h_b, N);
    Matrix C(N, MatrixInitType::ZERO);

    C.matrix_add(A, B);
    std::vector<int> result = C.export_to_host();
    verify_result_add(h_a, h_b, result, N);

    int factor = 4;
    C.matrix_scale(A, factor);
    result = C.export_to_host();
    verify_result_scale(h_a, factor, result, N);

    C.matrix_mul(A, B);
    result = C.export_to_host();
    verify_result_mul(h_a, h_b, result, N);
    
    std::cout << "COMPLETED SUCCESSFULLY" << std::endl;

    return 0;
}