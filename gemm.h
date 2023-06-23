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


#endif //GEMM_H

#ifndef GEMM_IMPLEMENTATION
#define GEMM_IMPLEMENTATION

void vec1_avx_mul(float32* res, float32* mat1, float32* mat2, int mat_size)
{
    int AVX_SIZE = 8;  // AVX can process 8 floats at a time

    // Calculate the number of AVX chunks in the vector
    int num_avx_chunks = mat_size / AVX_SIZE;

    // Loop over each AVX chunk of the vectors
    for (int i = 0; i < num_avx_chunks; i++) {
        // Calculate the starting index for the current chunk
        int ii = i * AVX_SIZE;

        __m256 a = _mm256_load_ps(&mat1[ii]);
        __m256 b = _mm256_load_ps(&mat2[ii]);
        __m256 sum = _mm256_mul_ps(a, b);

        _mm256_store_ps(&res[ii], sum);
    }

    // Handle remaining elements with simple scalar multiplication
    int remaining_start = num_avx_chunks * AVX_SIZE;
    for (int i = remaining_start; i < mat_size; i++) {
        res[i] = mat1[i] * mat2[i];
    }
}


#endif //GEMM_IMPLEMENTATION