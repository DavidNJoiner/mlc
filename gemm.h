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
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <immintrin.h>

//#define BLOCK_SIZE 12

void vec1_avx_mul(float32* res, float32* mat1, float32* mat2, int mat_size);
void matmul(float32* res, float32* mat1, float32* mat2, int mat_size, int stride);

#endif //GEMM_H

#ifndef GEMM_IMPLEMENTATION
#define GEMM_IMPLEMENTATION

void vec1_avx_mul(float32* res, float32* mat1, float32* mat2, int mat_size)
{
    int CHAIN_SIZE = 8; // Process CHAIN_SIZE float32 numbers at a time

    // Calculate the number of chains in the vector
    int num_chains = mat_size / CHAIN_SIZE;

    printf("------------------ \n");
    printf("mat_size   : %d \n", mat_size);
    printf("chain_size : %d \n", CHAIN_SIZE);
    printf("num_chains : %d \n", num_chains);
    printf("------------------ \n");

    // Loop over each chain of the vectors
    for (int i = 0; i < num_chains; i++) {
        // Calculate the starting index for the current chain
        int ii = i * CHAIN_SIZE;
        int j = 0;

        // AVX can process 8 floats at a time, do this while we have enough floats left
        for (; j < CHAIN_SIZE - 7; j += 8) {
            __m256 a = _mm256_load_ps(&mat1[ii + j]);
            __m256 b = _mm256_load_ps(&mat2[ii + j]);
            __m256 sum = _mm256_mul_ps(a, b);
            _mm256_store_ps(&res[ii + j], sum);
        }

        // Handle remaining elements within this chain with simple scalar multiplication
        for (; j < CHAIN_SIZE; j++) {
            res[ii + j] = mat1[ii + j] * mat2[ii + j];
        }
    }

    // Handle remaining elements outside of complete chains with simple scalar multiplication
    int remaining_start = num_chains * CHAIN_SIZE;
    for (int i = remaining_start; i < mat_size; i++) {
        res[i] = mat1[i] * mat2[i];
    }
}


void matmul(float32* res, float32* mat1, float32* mat2, int mat_size, int stride)
{
    int BLOCK_SIZE = 12;
    // (mat_size / stride)

    // Calculate the number of blocks in each dimension
    int num_blocks = mat_size / BLOCK_SIZE;

    printf("------------------ \n");
    printf("mat_size   : %d \n", mat_size);
    printf("block_size : %d \n", BLOCK_SIZE);
    printf("num_block  : %d \n", num_blocks);
    printf("------------------ \n");

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
                for (int i2 = 0; i2 < BLOCK_SIZE; i2 += 12) {
                    for (int j2 = 0; j2 < BLOCK_SIZE; j2 += 12) {
                        __m256 sum = _mm256_setzero_ps();
                        for (int k2 = 0; k2 < BLOCK_SIZE; k2 += 3) {
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