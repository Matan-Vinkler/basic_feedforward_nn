#include <iostream>
#include <vector>
#include <algorithm>

#include <cublas_v2.h>

#include "inc/Matrix.cuh"
#include "inc/utils/verity_results.h"

int main()
{
    std::cout << "Starting..." << std::endl;
    srand((unsigned int)time(NULL));

    cublasHandle_t handle;
    cublasCreate(&handle);

    int NUM_ROWS_A = 512;
    int NUM_COLS_A = 128;

    int NUM_ROWS_B = NUM_ROWS_A;
    int NUM_COLS_B = NUM_COLS_A;

    int NUM_ROWS_C = NUM_COLS_A;
    int NUM_COLS_C = 256;

    std::vector<float> h_a(NUM_ROWS_A * NUM_COLS_A);
    std::vector<float> h_b(NUM_ROWS_B * NUM_COLS_B);
    std::vector<float> h_c(NUM_ROWS_C * NUM_COLS_C);

    std::generate(h_a.begin(), h_a.end(), []() { return (float)rand() / (float)(RAND_MAX / 1); });
    std::generate(h_b.begin(), h_b.end(), []() { return (float)rand() / (float)(RAND_MAX / 1); });
    std::generate(h_c.begin(), h_c.end(), []() { return (float)rand() / (float)(RAND_MAX / 1); });

    Matrix A(handle, h_a, NUM_ROWS_A, NUM_COLS_A);
    Matrix B(handle, h_b, NUM_ROWS_B, NUM_COLS_B);
    Matrix C(handle, h_c, NUM_ROWS_C, NUM_COLS_C);

    Matrix D1 = A + B;
    std::vector<float> result1 = D1.export_to_host();

    int scalar = 4;
    Matrix D2 = scalar * A;
    std::vector<float> result2 = D2.export_to_host();

    Matrix D3 = A * C;
    std::vector<float> result3 = D3.export_to_host();

    std::cout << "Calculated matrices, now verifying..." << std::endl;

    verify_result_add(h_a, h_b, result1, NUM_ROWS_A, NUM_COLS_A);
    verify_result_scale(h_a, scalar, result2, NUM_ROWS_A, NUM_COLS_A);
    verify_result_mul(h_a, h_c, result3, NUM_ROWS_A, NUM_COLS_A, NUM_COLS_C);
    
    std::cout << "COMPLETED SUCCESSFULLY" << std::endl;

    cublasDestroy(handle);

    return 0;
}