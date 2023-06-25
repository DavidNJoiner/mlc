#ifndef AVX_H_ 
#define AVX_H_

#include "dtype.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <immintrin.h>

void    vec1_avx_mul_float32(float32* res, float32* mat1, float32* mat2, int mat_size);
void    vec1_avx_mul_float64(float64* res, float64* mat1, float64* mat2, int mat_size);
void    vec1_avx_add_float32(float32* res, float32* mat1, int mat_size);
void    vec1_avx_add_float64(float64* res, float64* mat1, int mat_size);


#endif //AVX_H

#ifndef AVX_IMPLEMENTATION
#define AVX_IMPLEMENTATION

/*
    -------------------------------------------------------
    vec1_avx_mul_float32 : Multiply two 1D float32 vector using AVX intrinsics.
    -------------------------------------------------------
*/
void vec1_avx_mul_float32(float32* dst, float32* A, float32* B, int mat_size)
{
    int AVX_SIZE = 8;  // AVX can process 8 floats at a time
    int num_avx_chunks = mat_size / AVX_SIZE;

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
/*
    -------------------------------------------------------
    vec1_avx_mul_float32 : Multiply two 1D float64 vector using AVX intrinsics.
    -------------------------------------------------------
*/
void vec1_avx_mul_float64(float64* dst, float64* A, float64* B, int mat_size)
{
    int AVX_SIZE = 4;  // AVX can process 4 double at a time
    int num_avx_chunks = mat_size / AVX_SIZE;

    for (int i = 0; i < num_avx_chunks; i++) {
        // Calculate the starting index for the current chunk
        int ii = i * AVX_SIZE;

        __m256d a = _mm256_load_pd(&A[ii]);
        __m256d b = _mm256_load_pd(&B[ii]);
        __m256d sum = _mm256_mul_pd(a, b);

        _mm256_store_pd(&dst[ii], sum);
    }

    // Handle remaining elements with simple scalar multiplication
    int remaining_start = num_avx_chunks * AVX_SIZE;
    for (int i = remaining_start; i < mat_size; i++) {
        dst[i] = A[i] * B[i];
    }
}
/*
    -------------------------------------------------------
    vec1_avx_add_float32 : Add two 1D float32 vector using AVX intrinsics.
    -------------------------------------------------------
*/
void vec1_avx_add_float32(float32* dst, float32* A, int mat_size)
{
    int AVX_SIZE = 8;  // AVX can process 8 floats at a time
    int num_avx_chunks = mat_size / AVX_SIZE;

    for (int i = 0; i < num_avx_chunks; i++) {
        // Calculate the starting index for the current chunk
        int ii = i * AVX_SIZE;

        __m256 a = _mm256_load_ps(&A[ii]);
        __m256 dst_chunk = _mm256_load_ps(&dst[ii]);
        __m256 sum = _mm256_add_ps(dst_chunk, a);

        _mm256_store_ps(&dst[ii], sum);
    }

    // Handle remaining elements with simple scalar addition
    int remaining_start = num_avx_chunks * AVX_SIZE;
    for (int i = remaining_start; i < mat_size; i++) {
        dst[i] += A[i];
    }
}
/*
    -------------------------------------------------------
    vec1_avx_add_float64 : Add two 1D float64 vector using AVX intrinsics.
    -------------------------------------------------------
*/
void vec1_avx_add_float64(float64* dst, float64* A, int mat_size)
{
    int AVX_SIZE = 4;  // AVX can process 4 double at a time
    int num_avx_chunks = mat_size / AVX_SIZE;

    for (int i = 0; i < num_avx_chunks; i++) {
        // Calculate the starting index for the current chunk
        int ii = i * AVX_SIZE;

        __m256d a = _mm256_load_pd(&A[ii]);
        __m256d dst_chunk = _mm256_load_pd(&dst[ii]);
        __m256d sum = _mm256_add_pd(dst_chunk, a);

        _mm256_store_pd(&dst[ii], sum);
    }

    // Handle remaining elements with simple scalar addition
    int remaining_start = num_avx_chunks * AVX_SIZE;
    for (int i = remaining_start; i < mat_size; i++) {
        dst[i] += A[i];
    }
}



#endif //AVX_IMPLEMENTATION