//
// Created by depaulsmiller on 11/15/20.
//
#include <cassert>
#include <chrono>
#include <iostream>
#include <immintrin.h>

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

    }

    Matrix(int rows, int columns, float *d) : rows_(rows), columns_(columns), data(d) {

    }


    ~Matrix() {
        delete[] data;
    }

    Matrix multiply(const Matrix &rhs, double &time) {

        assert(columns_ == rhs.rows_);

        float *result = new float[rows_ * rhs.columns_];

        for (int i = 0; i < rows_ * rhs.columns_; i++) {
            result[i] = 0.0f;
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < rows_; i++) {
            for (int j = 0; j < rhs.columns_; j++) {
                for (int k = 0; k < columns_; k++) {
                    result[i + j * rows_] += data[i + k * rows_] * rhs.data[k + j * columns_];
                }
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        time = std::chrono::duration<double>(end - start).count();

        return Matrix(rows_, rhs.columns_, result);
    }

    Matrix multiply_loopflip(const Matrix &rhs, double &time) {

        assert(columns_ == rhs.rows_);

        float *result = new float[rows_ * rhs.columns_];

        for (int i = 0; i < rows_ * rhs.columns_; i++) {
            result[i] = 0.0f;
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < rhs.columns_; j++) {
            for (int i = 0; i < rows_; i++) {
                for (int k = 0; k < columns_; k++) {
                    result[i + j * rows_] += data[i + k * rows_] * rhs.data[k + j * columns_];
                }
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        time = std::chrono::duration<double>(end - start).count();

        return Matrix(rows_, rhs.columns_, result);
    }

    Matrix multiply_tiling(const Matrix &rhs, double &time) {

        assert(columns_ == rhs.rows_);

        float *result = new float[rows_ * rhs.columns_];

        for (int i = 0; i < rows_ * rhs.columns_; i++) {
            result[i] = 0.0f;
        }

        const int jtiling = 4;
        const int ktiling = 4;

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < rows_; i++) {
            for (int j = 0; j < rhs.columns_; j += jtiling) {
                for (int k = 0; k < columns_; k += ktiling) {
                    int i_plus_j_time_rows_ = i + j * rows_;
                    int i_plus_k_time_rows_ = i + k * rows_;
                    int j_time_columns_ = j * columns_;
                    for (int jb = 0; jb < std::min(jtiling, rhs.columns_ - j); jb++) {
                        for (int kb = 0; kb < std::min(ktiling, columns_ - k); kb++) {
                            result[i_plus_j_time_rows_ + jb * rows_] += data[i_plus_k_time_rows_ + kb * rows_] *
                                                                        rhs.data[k + kb + j_time_columns_ +
                                                                                 jb * columns_];
                        }
                    }
                }
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        time = std::chrono::duration<double>(end - start).count();

        return Matrix(rows_, rhs.columns_, result);
    }

    Matrix multiply_loopflip_tiling(const Matrix &rhs, double &time) {

        assert(columns_ == rhs.rows_);

        float *result = new float[rows_ * rhs.columns_];

        for (int i = 0; i < rows_ * rhs.columns_; i++) {
            result[i] = 0.0f;
        }

        const int itiling = 4;
        const int ktiling = 4;

        auto start = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < rhs.columns_; j++) {
            for (int i = 0; i < rows_; i += itiling) {
                for (int k = 0; k < columns_; k += ktiling) {
                    for (int ib = 0; ib < std::min(itiling, rows_ - i); ib++) {
                        for (int kb = 0; kb < std::min(ktiling, columns_ - k); kb++) {
                            result[i + j * rows_ + ib] +=
                                    data[i + k * rows_ + ib + kb * rows_] * rhs.data[kb + k + j * columns_];
                        }
                    }
                }
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        time = std::chrono::duration<double>(end - start).count();

        return Matrix(rows_, rhs.columns_, result);
    }

    Matrix multiply_loopflip_tiling_128avx(const Matrix &rhs, double &time) {

        assert(columns_ == rhs.rows_);

        float *result = new float[rows_ * rhs.columns_];

        for (int i = 0; i < rows_ * rhs.columns_; i++) {
            result[i] = 0.0f;
        }

        const int itiling = sizeof(__m128) / sizeof(float);
        const int ktiling = 4;

        auto start = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < rhs.columns_; j++) {
            for (int i = 0; i < rows_; i += itiling) {
                for (int k = 0; k < columns_; k += ktiling) {
                    __m128 resVec = _mm_loadu_ps(&result[i + j * rows_]);

                    for (int kb = 0; kb < std::min(ktiling, columns_ - k); kb++) {

                        __m128 AVec = _mm_loadu_ps(&data[i + k * rows_ + kb * rows_]);
                        __m128 BVec = _mm_loadu_ps(&rhs.data[kb + k + j * columns_]);

                        resVec = _mm_fmadd_ps(AVec, BVec, resVec);

                    }
                    _mm_storeu_ps(&result[i + j * rows_], resVec);
                }
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        time = std::chrono::duration<double>(end - start).count();

        return Matrix(rows_, rhs.columns_, result);
    }

    Matrix multiply_loopflip_tiling_256avx(const Matrix &rhs, double &time) {

        assert(columns_ == rhs.rows_);

        float *result = new float[rows_ * rhs.columns_];

        for (int i = 0; i < rows_ * rhs.columns_; i++) {
            result[i] = 0.0f;
        }

        const int itiling = sizeof(__m256) / sizeof(float);
        const int ktiling = 4;

        auto start = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < rhs.columns_; j++) {
            for (int i = 0; i < rows_; i += itiling) {
                for (int k = 0; k < columns_; k += ktiling) {
                    __m256 resVec = _mm256_loadu_ps(&result[i + j * rows_]);

                    for (int kb = 0; kb < std::min(ktiling, columns_ - k); kb++) {

                        __m256 AVec = _mm256_loadu_ps(&data[i + k * rows_ + kb * rows_]);
                        __m256 BVec = _mm256_loadu_ps(&rhs.data[kb + k + j * columns_]);

                        resVec = _mm256_fmadd_ps(AVec, BVec, resVec);

                    }

                    _mm256_storeu_ps(&result[i + j * rows_], resVec);
                }
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        time = std::chrono::duration<double>(end - start).count();

        return Matrix(rows_, rhs.columns_, result);
    }



    IndexObject operator[](int i) {
        return IndexObject(i, rows_, columns_, data);
    }

private:
    int rows_;
    int columns_;
    float *data;
};

int main() {

    int rowsA = 128;
    int columns = 128;
    int columnsB = 128;

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

    Matrix C2 = A.multiply_tiling(B, time);

    std::cout << floatingPointOps / time / 1e9 << " GFLOPS" << std::endl;

    Matrix C3 = A.multiply_loopflip_tiling(B, time);

    std::cout << floatingPointOps / time / 1e9 << " GFLOPS" << std::endl;

    Matrix C4 = A.multiply_loopflip_tiling_128avx(B, time);

    std::cout << floatingPointOps / time / 1e9 << " GFLOPS" << std::endl;

    Matrix C5 = A.multiply_loopflip_tiling_256avx(B, time);

    std::cout << floatingPointOps / time / 1e9 << " GFLOPS" << std::endl;

    return 0;

}