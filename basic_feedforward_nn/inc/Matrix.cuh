#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <algorithm>
#include <cassert>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>

enum class MatrixInitType
{
	ZERO,
	RANDOM
};

class Matrix
{
public:
	Matrix(cublasHandle_t handle, int NUM_ROWS, int NUM_COLS, MatrixInitType init_type);
	Matrix(cublasHandle_t handle, std::vector<float> data, int NUM_ROWS, int NUM_COLS);
	Matrix(const Matrix& other);
	~Matrix();

	void matrix_add(Matrix& matrix_a, Matrix& matrix_b);
	void matrix_scale(Matrix& matrix_a, float scalar);
	void matrix_mul(Matrix& matrix_a, Matrix& matrix_b);

	void print_shape();

	friend Matrix operator+(Matrix& A, Matrix& B);
	friend Matrix operator*(Matrix& A, float scalar);
	friend Matrix operator*(float scalar, Matrix& A);
	friend Matrix operator*(Matrix& A, Matrix& B);

	std::vector<float> export_to_host();

private:
	float* dev_data;

	int NUM_ROWS;
	int NUM_COLS;

	cublasHandle_t handle;
};

#endif // !MATRIX_H