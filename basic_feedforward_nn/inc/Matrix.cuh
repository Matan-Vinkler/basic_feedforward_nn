#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <algorithm>
#include <cassert>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "utils/matrix_ops.cuh"

#define DEFAULT_THREADS_NUM 32

enum class MatrixInitType
{
	ZERO,
	RANDOM
};

class Matrix
{
public:
	Matrix(int N, MatrixInitType init_type);
	Matrix(std::vector<float> data, int N);
	~Matrix();

	void matrix_add(Matrix& matrix_a, Matrix& matrix_b);
	void matrix_scale(Matrix& matrix_a, int factor_scale);
	void matrix_mul(Matrix& matrix_a, Matrix& matrix_b);

	std::vector<float> export_to_host();

private:
	float* dev_data;
	int N;
};

#endif // !MATRIX_H
