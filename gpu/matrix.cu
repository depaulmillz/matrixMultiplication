//
// Created by depaulsmiller on 11/16/20.
//
#include <cassert>
#include <chrono>
#include <iostream>
#include <cublas_v2.h>

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
const int BLOCKSIZE2 = 48;

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

        C[IDX2C(i, j, rowsA)] = reg;
    }
}

__global__ void
multiply_transpose_tile_kern(int rowsA, int cols, int colsB, const float *__restrict__ A, const float *__restrict__ B,
                             float *__restrict__ C) {

    int i = (int) (threadIdx.x + blockIdx.x * BLOCKSIZE);
    int j = (int) (threadIdx.y + blockIdx.y * BLOCKSIZE);

    __shared__ float subAT[BLOCKSIZE * BLOCKSIZE];
    __shared__ float subB[BLOCKSIZE * BLOCKSIZE];

    float reg = 0.0;
    if (i < rowsA && j < colsB) {
        // let initial_reg = reg;
        int w = 0;

        subAT[IDX2C(threadIdx.y, threadIdx.x, BLOCKSIZE)] = A[IDX2C(i, w * BLOCKSIZE + threadIdx.y, rowsA)];
        subB[IDX2C(threadIdx.x, threadIdx.y, BLOCKSIZE)] = B[IDX2C(w * BLOCKSIZE + threadIdx.x, j, cols)];
        __syncthreads();
        for (int k = 0; k < min(BLOCKSIZE, cols - w * BLOCKSIZE); k++) {
            reg = fma(subAT[IDX2C(k, threadIdx.x, BLOCKSIZE)], subB[IDX2C(k, threadIdx.y, BLOCKSIZE)], reg);
        }

        for (w = 1; w < gridDim.y; w++) {
            __syncthreads();
            subAT[IDX2C(threadIdx.y, threadIdx.x, BLOCKSIZE)] = A[IDX2C(i, w * BLOCKSIZE + threadIdx.y, rowsA)];
            subB[IDX2C(threadIdx.x, threadIdx.y, BLOCKSIZE)] = B[IDX2C(w * BLOCKSIZE + threadIdx.x, j, cols)];
            __syncthreads();
            for (int k = 0; k < min(BLOCKSIZE, cols - w * BLOCKSIZE); k++) {
                reg = fma(subAT[IDX2C(k, threadIdx.x, BLOCKSIZE)], subB[IDX2C(k, threadIdx.y, BLOCKSIZE)], reg);
            }
        }
        C[IDX2C(i, j, rowsA)] = reg;
    }
}

#define WPLUS1 w + 1

__global__ void
multiply_transpose_overlap_tile_kern(int rowsA, int cols, int colsB, const float *__restrict__ A,
                                     const float *__restrict__ B,
                                     float *__restrict__ C) {

    int i = (int) (threadIdx.x + blockIdx.x * BLOCKSIZE2);
    int j = (int) (threadIdx.y + blockIdx.y * BLOCKSIZE2);

    __shared__ float subAT1[BLOCKSIZE2 * BLOCKSIZE2];
    __shared__ float subB1[BLOCKSIZE2 * BLOCKSIZE2];
    __shared__ float subAT2[BLOCKSIZE2 * BLOCKSIZE2];
    __shared__ float subB2[BLOCKSIZE2 * BLOCKSIZE2];

    float reg = 0.0;
    if (i < rowsA && j < colsB) {
        // let initial_reg = reg;
        for (int w = 0; w < gridDim.y; w += 2) {

            if (WPLUS1 < gridDim.y) {

                subAT2[IDX2C(threadIdx.y, threadIdx.x, BLOCKSIZE2)] = A[IDX2C(i, WPLUS1 * BLOCKSIZE2 + threadIdx.y,
                                                                              rowsA)];
                subB2[IDX2C(threadIdx.x, threadIdx.y, BLOCKSIZE2)] = B[IDX2C(WPLUS1 * BLOCKSIZE2 + threadIdx.x, j,
                                                                             cols)];

            }
            __syncthreads();

            subAT1[IDX2C(threadIdx.y, threadIdx.x, BLOCKSIZE2)] = A[IDX2C(i, w * BLOCKSIZE2 + threadIdx.y, rowsA)];

            subB1[IDX2C(threadIdx.x, threadIdx.y, BLOCKSIZE2)] = B[IDX2C(w * BLOCKSIZE2 + threadIdx.x, j, cols)];

            if (WPLUS1 < gridDim.y) {
                for (int k = 0; k < min(BLOCKSIZE2, cols - WPLUS1 * BLOCKSIZE2); k++) {
                    reg = fma(subAT2[IDX2C(k, threadIdx.x, BLOCKSIZE2)], subB2[IDX2C(k, threadIdx.y, BLOCKSIZE2)], reg);
                }
            }

            __syncthreads();

            for (int k = 0; k < min(BLOCKSIZE, cols - w * BLOCKSIZE2); k++) {
                reg = fma(subAT1[IDX2C(k, threadIdx.x, BLOCKSIZE2)], subB1[IDX2C(k, threadIdx.y, BLOCKSIZE2)], reg);
            }

        }

        C[IDX2C(i, j, rowsA)] = reg;
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

    Matrix multiply_tile_opt(const Matrix &rhs, double &time) {

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
        multiply_transpose_tile_kern<<<blocksInGrid, blockSize>>>(rows_, columns_, rhs.columns_, data_k, rhs.data_k,
                                                                  result_k);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            exit(1);
        }
        auto end = std::chrono::high_resolution_clock::now();

        cudaMemcpy(result, result_k, sizeof(float) * rows_ * rhs.columns_, cudaMemcpyDeviceToHost);


        time = std::chrono::duration<double>(end - start).count();

        return Matrix(rows_, rhs.columns_, result, result_k);
    }

    Matrix multiply_tile_opt2(const Matrix &rhs, double &time) {

        assert(columns_ == rhs.rows_);

        float *result = new float[rows_ * rhs.columns_];

        float *result_k;

        cudaMalloc(&result_k, sizeof(float) * rows_ * rhs.columns_);

        for (int i = 0; i < rows_ * rhs.columns_; i++) {
            result[i] = 0.0f;
        }

        cudaMemcpy(data_k, data, sizeof(float) * rows_ * columns_, cudaMemcpyHostToDevice);
        cudaMemcpy(rhs.data_k, rhs.data, sizeof(float) * rhs.rows_ * rhs.columns_, cudaMemcpyHostToDevice);

        dim3 blockSize(BLOCKSIZE2, BLOCKSIZE2);
        dim3 blocksInGrid((rows_ + BLOCKSIZE2 - 1) / BLOCKSIZE2, (rhs.columns_ + BLOCKSIZE2 - 1) / BLOCKSIZE2);

        auto start = std::chrono::high_resolution_clock::now();
        multiply_transpose_overlap_tile_kern<<<blocksInGrid, blockSize>>>(rows_, columns_, rhs.columns_, data_k,
                                                                          rhs.data_k,
                                                                          result_k);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            exit(1);
        }
        auto end = std::chrono::high_resolution_clock::now();

        cudaMemcpy(result, result_k, sizeof(float) * rows_ * rhs.columns_, cudaMemcpyDeviceToHost);


        time = std::chrono::duration<double>(end - start).count();

        return Matrix(rows_, rhs.columns_, result, result_k);
    }


    Matrix multiply_cublas(const Matrix &rhs, double &time) {

        assert(columns_ == rhs.rows_);

        cublasHandle_t handle;

        cublasStatus_t stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf("CUBLAS initialization failed\n");
            exit(1);
        }

        float *result = new float[rows_ * rhs.columns_];

        float *result_k;

        cudaMalloc(&result_k, sizeof(float) * rows_ * rhs.columns_);

        for (int i = 0; i < rows_ * rhs.columns_; i++) {
            result[i] = 0.0f;
        }

        cudaMemcpy(data_k, data, sizeof(float) * rows_ * columns_, cudaMemcpyHostToDevice);
        cudaMemcpy(rhs.data_k, rhs.data, sizeof(float) * rhs.rows_ * rhs.columns_, cudaMemcpyHostToDevice);

        float alpha = 1.0;
        float beta = 0.0;


        auto start = std::chrono::high_resolution_clock::now();

        stat = cublasSgemm(handle,
                           CUBLAS_OP_N, CUBLAS_OP_N, rows_, rhs.columns_, columns_,
                           &alpha,
                           this->data_k, rows_,
                           rhs.data_k, columns_,
                           &beta,
                           result_k, rows_);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf("CUBLAS initialization failed\n");
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
    int rowsA = 4096;
    int columns = 4096;
    int columnsB = 4096;

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

    double time;

    auto C2 = A.multiply_tile(B, time);

    double baseline = time;

    std::cout << "Baseline: " << baseline * 1e6 << " (us)" << std::endl;

    //auto C3 = A.multiply_tile(B, time);

    //std::cout << baseline / time << "x with " << time * 1e6 << "us" << std::endl;

    auto C4 = A.multiply_tile_opt(B, time);

    std::cout << baseline / time << "x with " << time * 1e6 << "us" << std::endl;

    auto C5 = A.multiply_tile_opt2(B, time);

    std::cout << baseline / time << "x with " << time * 1e6 << "us" << std::endl;

    auto C6 = A.multiply_cublas(B, time);

    std::cout << baseline / time << "x with " << time * 1e6 << "us" << std::endl;

    return 0;

}