#ifndef UTILS_MATRIX_OPS_CUH
#define UTILS_MATRIX_OPS_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*
* Performs matrix multiplication between matrices a and b, storing the result in matrix c.
*
* Arguments:
*  - a: Pointer to the first input matrix.
*  - b: Pointer to the second input matrix.
*  - c: Pointer to the output matrix.
*  - N: Dimension size of the NxN matrices.
*
* Return:
*  - None.
*/
__global__ void matrixMul(const float* a, const float* b, float* c, int N);

/*
* Adds two matrices a and b, storing the result in matrix c.
*
* Arguments:
*  - a: Pointer to the first input matrix.
*  - b: Pointer to the second input matrix.
*  - c: Pointer to the output matrix.
*  - N: Dimension size of the NxN matrices.
*
* Return:
*  - None.
*/
__global__ void matrixAdd(const float* a, const float* b, float* c, int N);

/*
* Scales a matrix a by a given factor, storing the result in matrix c.
*
* Arguments:
*  - a: Pointer to the input matrix.
*  - factor: The scaling factor.
*  - c: Pointer to the output matrix.
*  - N: Dimension size of the NxN matrix.
*
* Return:
*  - None.
*/
__global__ void matrixScale(const float* a, int factor, float* c, int N);

#endif // !CUDA_MATRIX_OPS
