#include "../inc/Matrix.cuh"

Matrix::Matrix(int N, MatrixInitType init_type) : N(N), dev_data(NULL)
{
	std::vector<int> init_data_fill(N * N);

	switch (init_type)
	{
	case MatrixInitType::ZERO:
		std::fill(init_data_fill.begin(), init_data_fill.end(), 0);
		break;
	case MatrixInitType::RANDOM:
		std::generate(init_data_fill.begin(), init_data_fill.end(), []() {return rand() % 100; });
		break;
	default:
		break;
	}

	size_t bytes = N * N * sizeof(int);

	cudaMalloc(&dev_data, bytes);
	cudaMemcpy(dev_data, init_data_fill.data(), bytes, cudaMemcpyHostToDevice);
}

Matrix::Matrix(std::vector<int> data, int N) : N(N), dev_data(NULL)
{
	size_t bytes = N * N * sizeof(int);

	cudaMalloc(&dev_data, bytes);
	cudaMemcpy(dev_data, data.data(), bytes, cudaMemcpyHostToDevice);
}

Matrix::~Matrix()
{
	cudaFree(dev_data);
}

void Matrix::matrix_add(Matrix& matrix_a, Matrix& matrix_b)
{
	assert(matrix_a.N == matrix_b.N);
	assert(matrix_a.N == N);

	int THREADS = DEFAULT_THREADS_NUM;
	int BLOCKS = N / THREADS;

	dim3 threads(THREADS, THREADS);
	dim3 blocks(BLOCKS, BLOCKS);

	matrixAdd << < blocks, threads >> > (matrix_a.dev_data, matrix_b.dev_data, dev_data, N);
}

void Matrix::matrix_scale(Matrix& matrix_a, int factor_scale)
{
	assert(matrix_a.N == N);

	int THREADS = DEFAULT_THREADS_NUM;
	int BLOCKS = N / THREADS;

	dim3 threads(THREADS, THREADS);
	dim3 blocks(BLOCKS, BLOCKS);

	matrixScale << < blocks, threads >> > (matrix_a.dev_data, factor_scale, dev_data, N);
}

void Matrix::matrix_mul(Matrix& matrix_a, Matrix& matrix_b)
{
	assert(matrix_a.N == matrix_b.N);
	assert(matrix_a.N == N);

	int THREADS = DEFAULT_THREADS_NUM;
	int BLOCKS = N / THREADS;

	dim3 threads(THREADS, THREADS);
	dim3 blocks(BLOCKS, BLOCKS);

	matrixMul << < blocks, threads >> > (matrix_a.dev_data, matrix_b.dev_data, dev_data, N);
}

std::vector<int> Matrix::export_to_host()
{
	std::vector<int> h_data(N * N);
	size_t bytes = N * N * sizeof(int);

	cudaMemcpy(h_data.data(), dev_data, bytes, cudaMemcpyDeviceToHost);

	return h_data;
}
