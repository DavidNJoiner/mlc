/* gemm.h
* a c program to multuply two matrices of size n. Optimized for Skylake-X CPU.

The BLOCK_SIZE constant determines the size of the blocks that we will use to partition the matrices.
Block multiplication done using AVX-optimized instructions.
Within each block multiplication, we further partition the blocks into sub-blocks of size 8x8
and use an AVX-optimized loop to perform the multiplication.
*/

#ifndef GEMM_H_ 
#define GEMM_H_

#include "dtype.h"
#include <time.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <immintrin.h>

#define BLOCK_SIZE 16

uint64_t nanos();
void matmul(float32* res, float32* mat1, float32* mat2, int mat_size);

#endif //GEMM_H

#ifndef GEMM_IMPLEMENTATION
#define GEMM_IMPLEMENTATION

uint64_t nanos(){
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC, &start);
    return (uint64_t)start.tv_sec*1000000000 + (uint64_t)start.tv_nsec;
}

void matmul(float32* res, float32* mat1, float32* mat2, int mat_size)
{
    // Calculate the number of blocks in each dimension
    int num_blocks = mat_size / BLOCK_SIZE;

    #pragma omp parallel for collapse(3) num_threads(16)
    // Loop over each block of the result matrix
    for (int i = 0; i < num_blocks; i++) {
        for (int j = 0; j < num_blocks; j++) {
            // Loop over each block of the input matrices
            for (int k = 0; k < num_blocks; k++) {
                // Calculate the starting indices for the current block
                int ii = i * BLOCK_SIZE;
                int jj = j * BLOCK_SIZE;
                int kk = k * BLOCK_SIZE;

                // Multiply the current blocks using AVX-optimized instructions
                for (int i2 = 0; i2 < BLOCK_SIZE; i2 += 8) {
                    for (int j2 = 0; j2 < BLOCK_SIZE; j2 += 8) {
                        __m256 sum = _mm256_setzero_ps();
                        for (int k2 = 0; k2 < BLOCK_SIZE; k2 += 8) {
                            __m256 a = _mm256_load_ps(&mat1[(ii + i2) * mat_size + kk + k2]);
                            __m256 b = _mm256_load_ps(&mat2[(kk + k2) * mat_size + jj + j2]);
                            sum = _mm256_fmadd_ps(a, b, sum);
                        }
                        _mm256_store_ps(&res[(ii + i2) * mat_size + jj + j2], sum);
                    }
                }
            }
        }
    }
}

#endif //GEMM_IMPLEMENTATION