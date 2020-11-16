//
// Created by depaulsmiller on 11/16/20.
//
#include <cassert>
#include <chrono>
#include <iostream>

#define IDX2C(i, j, ld) (((j)*(ld))+(i))

__global__ void multiply_kern(int rowsA, int cols, int colsB, const float *__restrict__ A, const float *__restrict__ B,
                              float *__restrict__ C) {

    int i = (int) (threadIdx.x + blockIdx.x * blockDim.x);
    int j = (int) (threadIdx.y + blockIdx.y * blockDim.y);

    float reg = 0.0;

    if (i < rowsA && j < colsB) {
        for (int k = 0; k < cols; k++) {
            reg = fma(A[IDX2C(i, k, rowsA)], B[IDX2C(k, j, cols)], reg);
        }
        C[IDX2C(i, j, rowsA)] = reg;
    }

}

const int BLOCKSIZE = 64;

__global__ void
multiply_tile_kern(int rowsA, int cols, int colsB, const float *__restrict__ A, const float *__restrict__ B,
                   float *__restrict__ C) {

    int i = (int) (threadIdx.x + blockIdx.x * BLOCKSIZE);
    int j = (int) (threadIdx.y + blockIdx.y * BLOCKSIZE);

    __shared__ float subA[BLOCKSIZE * BLOCKSIZE];
    __shared__ float subB[BLOCKSIZE * BLOCKSIZE];

    float reg = 0.0;
    if (i < rowsA && j < colsB) {
        // let initial_reg = reg;
        for (int w = 0; w < gridDim.y; w++) {

            subA[IDX2C(threadIdx.x, threadIdx.y, BLOCKSIZE)] = A[IDX2C(i, w * BLOCKSIZE + threadIdx.y, rowsA)];
            subB[IDX2C(threadIdx.x, threadIdx.y, BLOCKSIZE)] = B[IDX2C(w * BLOCKSIZE + threadIdx.x, j, cols)];
            __syncthreads();

            // let reg_prev_loop = reg
            for (int k = 0; k < min(BLOCKSIZE, cols - w * BLOCKSIZE); k++) {
                // subA(threadIdx.x, k) is A(i, w * BLOCKSIZE + k)
                // subB(k, threadIdx.y) is B(w * BLOCKSIZE + k, j)

                // let reg_prev = reg
                reg = fma(subA[IDX2C(threadIdx.x, k, BLOCKSIZE)], subB[IDX2C(k, threadIdx.y, BLOCKSIZE)], reg);
                // reg = reg_prev + A(i, w * BLOCKSIZE + k) * B(w * BLOCKSIZE + k, j)
            }
            // reg = reg_prev_loop + sum k from 0, blocksize - 1 A(i, w * BLOCKSIZE + k) * B(w * BLOCKSIZE + k, j)
            __syncthreads();
        }
        // reg = initial_reg + sum w from 0 to gridDim.y - 1 ( sum k from 0, blocksize - 1 A(i, w * BLOCKSIZE + k) * B(w * BLOCKSIZE + k, j) )
        // this means reg holds C(i,j)

        C[IDX2C(i,j,rowsA)] = reg;
    }
}


class IndexObject {
public:
    IndexObject(int row_idx, int rows, int columns, float *data) : row_idx_(row_idx), rows_(rows), columns_(columns),
                                                                   data_(data) {
    }

    ~IndexObject() {

    }

    float &operator[](int i) {
        return data_[row_idx_ + rows_ * i];
    }

private:
    int row_idx_;
    int rows_;
    int columns_;
    float *data_;
};

class Matrix {
public:
    Matrix(int rows, int columns) : rows_(rows), columns_(columns), data(new float[rows * columns]) {
        cudaMalloc(&data_k, sizeof(float) * rows * columns);
    }

    Matrix(int rows, int columns, float *d, float *d_k) : rows_(rows), columns_(columns), data(d), data_k(d_k) {

    }


    ~Matrix() {
        delete[] data;
    }

    Matrix multiply(const Matrix &rhs, double &time) {

        assert(columns_ == rhs.rows_);

        float *result = new float[rows_ * rhs.columns_];

        float *result_k;

        cudaMalloc(&result_k, sizeof(float) * rows_ * rhs.columns_);

        for (int i = 0; i < rows_ * rhs.columns_; i++) {
            result[i] = 0.0f;
        }

        cudaMemcpy(data_k, data, sizeof(float) * rows_ * columns_, cudaMemcpyHostToDevice);
        cudaMemcpy(rhs.data_k, rhs.data, sizeof(float) * rhs.rows_ * rhs.columns_, cudaMemcpyHostToDevice);

        dim3 blockSize(32, 32);
        dim3 blocksInGrid((rows_ + 31) / 32, (rhs.columns_ + 31) / 32);

        auto start = std::chrono::high_resolution_clock::now();
        multiply_kern<<<blocksInGrid, blockSize>>>(rows_, columns_, rhs.columns_, data_k, rhs.data_k, result_k);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            exit(1);
        }
        auto end = std::chrono::high_resolution_clock::now();

        cudaMemcpy(result, result_k, sizeof(float) * rows_ * rhs.columns_, cudaMemcpyDeviceToHost);


        time = std::chrono::duration<double>(end - start).count();

        return Matrix(rows_, rhs.columns_, result, result_k);
    }

    Matrix multiply_tile(const Matrix &rhs, double &time) {

        assert(columns_ == rhs.rows_);

        float *result = new float[rows_ * rhs.columns_];

        float *result_k;

        cudaMalloc(&result_k, sizeof(float) * rows_ * rhs.columns_);

        for (int i = 0; i < rows_ * rhs.columns_; i++) {
            result[i] = 0.0f;
        }

        cudaMemcpy(data_k, data, sizeof(float) * rows_ * columns_, cudaMemcpyHostToDevice);
        cudaMemcpy(rhs.data_k, rhs.data, sizeof(float) * rhs.rows_ * rhs.columns_, cudaMemcpyHostToDevice);

        dim3 blockSize(BLOCKSIZE, BLOCKSIZE);
        dim3 blocksInGrid((rows_ + BLOCKSIZE - 1) / BLOCKSIZE, (rhs.columns_ + BLOCKSIZE - 1) / BLOCKSIZE);

        auto start = std::chrono::high_resolution_clock::now();
        multiply_tile_kern<<<blocksInGrid, blockSize>>>(rows_, columns_, rhs.columns_, data_k, rhs.data_k, result_k);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            exit(1);
        }
        auto end = std::chrono::high_resolution_clock::now();

        cudaMemcpy(result, result_k, sizeof(float) * rows_ * rhs.columns_, cudaMemcpyDeviceToHost);


        time = std::chrono::duration<double>(end - start).count();

        return Matrix(rows_, rhs.columns_, result, result_k);
    }

    IndexObject operator[](int i) {
        return IndexObject(i, rows_, columns_, data);
    }

private:
    int rows_;
    int columns_;
    float *data;
    float *data_k;
};

int main() {

    int rowsA = 1024;
    int columns = 1024;
    int columnsB = 1024;

    Matrix A(rowsA, columns);
    Matrix B(columns, columnsB);

    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < columns; j++) {
            A[i][j] = rand() / (float) RAND_MAX * 10000;
        }
    }

    for (int i = 0; i < columns; i++) {
        for (int j = 0; j < columnsB; j++) {
            B[i][j] = rand() / (float) RAND_MAX * 10000;
        }
    }

    unsigned floatingPointOps = 2 * columns * rowsA * columnsB;

    double time;

    Matrix C2 = A.multiply(B, time);

    std::cout << floatingPointOps / time / 1e12 << " TFLOPS" << std::endl;

    Matrix C3 = A.multiply_tile(B, time);

    std::cout << floatingPointOps / time / 1e12 << " TFLOPS" << std::endl;

    return 0;

}