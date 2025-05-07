#include "../../inc/utils/verity_results.h"

#include <iostream>

void verify_result_mul(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c, int NUM_ROWS, int NUM_COMMON, int NUM_COLS)
{
    for (int row = 0; row < NUM_ROWS; row++)
    {
        for (int col = 0; col < NUM_COLS; col++)
        {
            float tmp = 0.0f;
            for (int k = 0; k < NUM_COMMON; k++)
            {
                tmp += a[k * NUM_ROWS + row] * b[col * NUM_COMMON + k];
            }

            if (std::fabs(tmp - c[col * NUM_ROWS + row]) > 1e-4)
            {
                std::cout << "Mismatch at (" << row << "," << col << ")\n";
                printf("tmp = %.5f\n", tmp);
                printf("c[col * NUM_ROWS + row] = %.5f\n", c[col * NUM_ROWS + row]);
                assert(tmp == c[col * NUM_ROWS + row]);
            }
        }
    }
}

void verify_result_add(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c, int NUM_ROWS, int NUM_COLS)
{
    for (int i = 0; i < NUM_ROWS; i++)
    {
        for (int j = 0; j < NUM_COLS; j++)
        {
            float tmp = a[i * NUM_COLS + j] + b[i * NUM_COLS + j];

            if (tmp != c[i * NUM_COLS + j])
            {
                std::cout << "Mismatch at (" << i << "," << j << ")\n";
                std::cout << "tmp = " << tmp << "\n";
                std::cout << "c[row * NUM_COLS + col] = " << c[i * NUM_COLS + j] << "\n";
                assert(tmp == c[i * NUM_COLS + j]);
            }
        }
    }
}

void verify_result_scale(std::vector<float>& a, int scalar, std::vector<float>& c, int NUM_ROWS, int NUM_COLS)
{
    for (int i = 0; i < NUM_ROWS; i++)
    {
        for (int j = 0; j < NUM_COLS; j++)
        {
            float tmp = a[i * NUM_COLS + j] * scalar;

            if (tmp != c[i * NUM_COLS + j])
            {
                std::cout << "Mismatch at (" << i << "," << j << ")\n";
                std::cout << "tmp = " << tmp << "\n";
                std::cout << "c[row * NUM_COLS + col] = " << c[i * NUM_COLS + j] << "\n";
                assert(tmp == c[i * NUM_COLS + j]);
            }
        }
    }
}