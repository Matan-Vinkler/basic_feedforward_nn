#include "../../inc/utils/verity_results.h"

void verify_result_mul(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float tmp = 0;
            for (int k = 0; k < N; k++)
            {
                tmp += a[i * N + k] * b[k * N + j];
            }

            assert(tmp == c[i * N + j]);
        }
    }
}

void verify_result_add(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float tmp = a[i * N + j] + b[i * N + j];

            assert(tmp == c[i * N + j]);
        }
    }
}

void verify_result_scale(std::vector<float>& a, int factor, std::vector<float>& c, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float tmp = a[i * N + j] * factor;

            assert(tmp == c[i * N + j]);
        }
    }
}