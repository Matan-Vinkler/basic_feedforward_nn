#include "../inc/Matrix.cuh"

Matrix::Matrix(cublasHandle_t handle, int NUM_ROWS, int NUM_COLS, MatrixInitType init_type) : handle(handle), NUM_ROWS(NUM_ROWS), NUM_COLS(NUM_COLS), dev_data(NULL)
{
	std::vector<float> init_data_fill(NUM_ROWS * NUM_COLS);

	switch (init_type)
	{
	case MatrixInitType::ZERO:
		std::fill(init_data_fill.begin(), init_data_fill.end(), 0);
		break;
	case MatrixInitType::RANDOM:
		std::generate(init_data_fill.begin(), init_data_fill.end(), []() {return (float)rand() / (float)(RAND_MAX / 1); });
		break;
	default:
		break;
	}

	size_t bytes = NUM_ROWS * NUM_COLS * sizeof(float);

	cudaMalloc(&dev_data, bytes);
	cudaMemcpy(dev_data, init_data_fill.data(), bytes, cudaMemcpyHostToDevice);
}

Matrix::Matrix(cublasHandle_t handle, std::vector<float> data, int NUM_ROWS, int NUM_COLS) : handle(handle), NUM_ROWS(NUM_ROWS), NUM_COLS(NUM_COLS), dev_data(NULL)
{
	size_t bytes = NUM_ROWS * NUM_COLS * sizeof(float);

	cudaMalloc(&dev_data, bytes);
	cudaMemcpy(dev_data, data.data(), bytes, cudaMemcpyHostToDevice);
}

Matrix::~Matrix()
{
	cudaFree(dev_data);
}

void Matrix::matrix_add(Matrix& matrix_a, Matrix& matrix_b)
{
	assert(matrix_a.NUM_ROWS == matrix_b.NUM_ROWS && matrix_a.NUM_COLS == matrix_b.NUM_COLS);
	assert(matrix_a.NUM_ROWS == NUM_ROWS && matrix_a.NUM_COLS == NUM_COLS);

	const float alpha = 1.0f, beta = 1.0f;

	int lda = NUM_ROWS;
	int ldb = NUM_ROWS;
	int ldc = NUM_ROWS;

	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, NUM_ROWS, NUM_COLS, &alpha, matrix_a.dev_data, lda, &beta, matrix_b.dev_data, ldb, dev_data, ldc);
}

void Matrix::matrix_scale(Matrix& matrix_a, float scalar)
{
	assert(matrix_a.NUM_ROWS == NUM_ROWS && matrix_a.NUM_COLS == NUM_COLS);

	cudaMemcpy(dev_data, matrix_a.dev_data, NUM_ROWS * NUM_COLS * sizeof(float), cudaMemcpyDeviceToDevice);

	cublasSscal(handle, NUM_ROWS * NUM_COLS, &scalar, dev_data, 1);
}

void Matrix::matrix_mul(Matrix& matrix_a, Matrix& matrix_b)
{
	assert(matrix_a.NUM_COLS == matrix_b.NUM_ROWS);
	assert(matrix_a.NUM_ROWS == NUM_ROWS && matrix_b.NUM_COLS == NUM_COLS);

	int m = matrix_a.NUM_ROWS;
	int k = matrix_a.NUM_COLS;
	int n = matrix_b.NUM_COLS;

	const float alpha = 1.0f;
	const float beta = 0.0f;

	int lda = m;
	int ldb = k;
	int ldc = m;

	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, matrix_a.dev_data, lda, matrix_b.dev_data, ldb, &beta, dev_data, ldc);
}

std::vector<float> Matrix::export_to_host()
{
	std::vector<float> h_data(NUM_ROWS * NUM_COLS);
	size_t bytes = NUM_ROWS * NUM_COLS * sizeof(float);

	cudaMemcpy(h_data.data(), dev_data, bytes, cudaMemcpyDeviceToHost);

	return h_data;
}
