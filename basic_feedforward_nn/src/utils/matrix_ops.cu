#include "../../inc/utils/matrix_ops.cuh"

__global__ void matrixMul(const int* a, const int* b, int* c, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    c[row * N + col] = 0;
    for (int k = 0; k < N; k++)
    {
        c[row * N + col] += a[row * N + k] * b[k * N + col];
    }
}

__global__ void matrixAdd(const int* a, const int* b, int* c, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    c[row * N + col] = a[row * N + col] + b[row * N + col];
}

__global__ void matrixScale(const int* a, int factor, int* c, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    c[row * N + col] = a[row * N + col] * factor;
}