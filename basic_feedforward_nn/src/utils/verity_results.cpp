#include "../../inc/utils/verity_results.h"

void verify_result_mul(std::vector<int>& a, std::vector<int>& b, std::vector<int>& c, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int tmp = 0;
            for (int k = 0; k < N; k++)
            {
                tmp += a[i * N + k] * b[k * N + j];
            }

            assert(tmp == c[i * N + j]);
        }
    }
}

void verify_result_add(std::vector<int>& a, std::vector<int>& b, std::vector<int>& c, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int tmp = a[i * N + j] + b[i * N + j];

            assert(tmp == c[i * N + j]);
        }
    }
}

void verify_result_scale(std::vector<int>& a, int factor, std::vector<int>& c, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int tmp = a[i * N + j] * factor;

            assert(tmp == c[i * N + j]);
        }
    }
}