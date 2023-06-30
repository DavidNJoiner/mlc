#include "avx.h"

/*  ----------------------------------------------------------------------------*/
/*  vec1_avx_mul_float16 : Multiply two 1D float16 vector using AVX intrinsics. */
/*  ----------------------------------------------------------------------------*/
void vec1_avx_mul_float16(float16* dst, float16* A, float16* B, int mat_size)
{
    int AVX_SIZE = 8;  // F16C can process 8 half floats at a time
    int num_avx_chunks = mat_size / AVX_SIZE;

    for (int i = 0; i < num_avx_chunks; i++) {
        // Calculate the starting index for the current chunk
        int ii = i * AVX_SIZE;

        // Load data
        __m128i a_half = _mm_load_si128((__m128i*)&A[ii]);
        __m128i b_half = _mm_load_si128((__m128i*)&B[ii]);

        // Convert to single precision
        __m256 a = _mm256_cvtph_ps(a_half);
        __m256 b = _mm256_cvtph_ps(b_half);

        // Perform multiplication
        __m256 product = _mm256_mul_ps(a, b);

        // Convert back to half precision
        __m128i product_half = _mm256_cvtps_ph(product, 0);

        // Store result
        _mm_store_si128((__m128i*)&dst[ii], product_half);
    }

    // Handle remaining elements with simple scalar multiplication
    int remaining_start = num_avx_chunks * AVX_SIZE;
    for (int i = remaining_start; i < mat_size; i++) {
        dst[i] = (float16)((float32)A[i] * (float32)B[i]);
    }
}
/*  ------------------------------------------------------------------------------*/
/*  vec1_avx_mul_float32 : Multiply two 1D float32 vector using AVX intrinsics.   */
/*  ------------------------------------------------------------------------------*/
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
/*  ------------------------------------------------------------------------------*/
/*  vec1_avx_mul_float32 : Multiply two 1D float64 vector using AVX intrinsics.   */
/*  ------------------------------------------------------------------------------*/
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
/*  ------------------------------------------------------------------------------*/
/*  vec1_avx_add_float16 : Add two 1D float16 vector using AVX intrinsics.        */
/*  ------------------------------------------------------------------------------*/
void vec1_avx_add_float16(float16* dst, float16* A, int mat_size)
{
    int AVX_SIZE = 8;  // F16C process 8 half floats at a time
    int num_avx_chunks = mat_size / AVX_SIZE;

    for (int i = 0; i < num_avx_chunks; i++) {
        // Calculate the starting index for the current chunk
        int ii = i * AVX_SIZE;

        // Load data
        __m128i a_half = _mm_load_si128((__m128i*)&A[ii]);
        __m128i dst_half = _mm_load_si128((__m128i*)&dst[ii]);

        // Convert to single precision ( AVX512_FP16 not widely supported yet)
        __m256 a = _mm256_cvtph_ps(a_half);
        __m256 dst_float = _mm256_cvtph_ps(dst_half);

        __m256 sum = _mm256_add_ps(dst_float, a);

        // Convert back to half precision
        __m128i sum_half = _mm256_cvtps_ph(sum, 0);

        _mm_store_si128((__m128i*)&dst[ii], sum_half);
    }

    // Handle remaining elements with simple scalar addition
    int remaining_start = num_avx_chunks * AVX_SIZE;
    for (int i = remaining_start; i < mat_size; i++) {
        // Assuming a software function to add half floats
        dst[i] = (float16)((float32)A[i] + (float32)dst[i]);
    }
}
/*  ------------------------------------------------------------------------------*/
/*  vec1_avx_add_float32 : Add two 1D float32 vector using AVX intrinsics.        */
/*  ------------------------------------------------------------------------------*/
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
/*  ------------------------------------------------------------------------------*/
/*  vec1_avx_add_float64 : Add two 1D float64 vector using AVX intrinsics.        */
/*  ------------------------------------------------------------------------------*/
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