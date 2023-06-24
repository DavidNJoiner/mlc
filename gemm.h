#ifndef GEMM_H_ 
#define GEMM_H_

#include "dtype.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <immintrin.h>


void    vec1_avx_mul(float32* res, float32* mat1, float32* mat2, int mat_size);

#endif //GEMM_H

#ifndef GEMM_IMPLEMENTATION
#define GEMM_IMPLEMENTATION

/*
    -------------------------------------------------------
    vec1_avx_mul : Multiply two 1D vectors and store the result in an other 1D vector.
    -------------------------------------------------------
*/
void vec1_avx_mul(float32* dst, float32* A, float32* B, int mat_size)
{
    int AVX_SIZE = 8;  // AVX can process 8 floats at a time
    int num_avx_chunks = mat_size / AVX_SIZE;

    // Loop over each AVX chunk of the vectors
    for (int i = 0; i < num_avx_chunks; i++) {
        // Calculate the starting index for the current chunk
        int ii = i * AVX_SIZE;

        __m256 a = _mm256_load_ps(&A[ii]);
        __m256 b = _mm256_load_ps(&B[ii]);
        __m256 sum = _mm256_mul_ps(a, b);

        _mm256_store_ps(&dst[ii], sum);
    }

    // Handle remaining elements with simple scalar multiplication
    int remaining_start = num_avx_chunks * AVX_SIZE;
    for (int i = remaining_start; i < mat_size; i++) {
        dst[i] = A[i] * B[i];
    }
}


#endif //GEMM_IMPLEMENTATION