#include "intel_intrinsics.h"

/*  ----------------------------------------------------------------------------*/
/*                                  AVX                                         */
/*  ----------------------------------------------------------------------------*/
/*  ----------------------------------------------------------------------------*/
/*  vec1_avx_mul_float16 : Multiply two 1D float16 vector using AVX intel_intrinsics. */
/*  ----------------------------------------------------------------------------*/
void vec1_avx_mul_float16(float16 *dst, float16 *A, float16 *B, int mat_size)
{
    uint16_t *dst_uint16 = (uint16_t *)dst;
    uint16_t *A_uint16 = (uint16_t *)A;
    uint16_t *B_uint16 = (uint16_t *)B;

    int AVX_SIZE = 8; // F16C can process 8 half floats at a time
    int num_avx_chunks = mat_size / AVX_SIZE;

    for (uint32_t i = 0; i < num_avx_chunks; i++)
    {
        // compute the starting index for the current chunk
        int ii = i * AVX_SIZE;

        // Load data
        __m128i a_half = _mm_load_si128((__m128i *)&A_uint16[ii]);
        __m128i b_half = _mm_load_si128((__m128i *)&B_uint16[ii]);

        // Convert to single precision
        __m256 a = _mm256_cvtph_ps(a_half);
        __m256 b = _mm256_cvtph_ps(b_half);

        // Perform multiplication
        __m256 product = _mm256_mul_ps(a, b);

        // Convert back to half precision
        __m128i product_half = _mm256_cvtps_ph(product, 0);

        // Store result
        _mm_store_si128((__m128i *)&dst_uint16[ii], product_half);
    }

    // Handle remaining elements with simple scalar multiplication
    int remaining_start = num_avx_chunks * AVX_SIZE;
    for (int i = remaining_start; i < mat_size; i++)
    {
        dst[i] = float16_from_float(A_uint16[i] * B_uint16[i]);
    }
}

/*  ------------------------------------------------------------------------------*/
/*  vec1_avx_mul_float32 : Multiply two 1D float32 vector using AVX intel_intrinsics.   */
/*  ------------------------------------------------------------------------------*/
void vec1_avx_mul_float32(float32 *dst, float32 *A, float32 *B, int mat_size)
{
    int AVX_SIZE = 8; // AVX can process 8 floats at a time
    int num_avx_chunks = mat_size / AVX_SIZE;

    for (uint32_t i = 0; i < num_avx_chunks; i++)
    {
        // compute the starting index for the current chunk
        int ii = i * AVX_SIZE;

        __m256 a = _mm256_load_ps(&A[ii]);
        __m256 b = _mm256_load_ps(&B[ii]);
        __m256 sum = _mm256_mul_ps(a, b);

        _mm256_store_ps(&dst[ii], sum);
    }

    // Handle remaining elements with simple scalar multiplication
    int remaining_start = num_avx_chunks * AVX_SIZE;
    for (int i = remaining_start; i < mat_size; i++)
    {
        dst[i] = A[i] * B[i];
    }
}
/*  ------------------------------------------------------------------------------*/
/*  vec1_avx_mul_float32 : Multiply two 1D float64 vector using AVX intel_intrinsics.   */
/*  ------------------------------------------------------------------------------*/
void vec1_avx_mul_float64(float64 *dst, float64 *A, float64 *B, int mat_size)
{
    int AVX_SIZE = 4; // AVX can process 4 double at a time
    int num_avx_chunks = mat_size / AVX_SIZE;

    for (uint32_t i = 0; i < num_avx_chunks; i++)
    {
        // compute the starting index for the current chunk
        int ii = i * AVX_SIZE;

        __m256d a = _mm256_load_pd(&A[ii]);
        __m256d b = _mm256_load_pd(&B[ii]);
        __m256d sum = _mm256_mul_pd(a, b);

        _mm256_store_pd(&dst[ii], sum);
    }

    // Handle remaining elements with simple scalar multiplication
    int remaining_start = num_avx_chunks * AVX_SIZE;
    for (int i = remaining_start; i < mat_size; i++)
    {
        dst[i] = A[i] * B[i];
    }
}
/*  ------------------------------------------------------------------------------*/
/*  vec1_avx_add_float16 : Add two 1D float16 vector using AVX intel_intrinsics.        */
/*  ------------------------------------------------------------------------------*/
void vec1_avx_add_float16(float16 *dst, float16 *A, int mat_size)
{
    uint16_t *dst_uint16 = (uint16_t *)dst;
    uint16_t *A_uint16 = (uint16_t *)A;

    int AVX_SIZE = 8; // F16C process 8 half floats at a time
    int num_avx_chunks = mat_size / AVX_SIZE;

    for (uint32_t i = 0; i < num_avx_chunks; i++)
    {
        // compute the starting index for the current chunk
        int ii = i * AVX_SIZE;

        // Load data
        __m128i a_half = _mm_load_si128((__m128i *)&A_uint16[ii]);
        __m128i dst_half = _mm_load_si128((__m128i *)&dst_uint16[ii]);

        // Convert to single precision ( AVX512_FP16 not widely supported yet)
        __m256 a = _mm256_cvtph_ps(a_half);
        __m256 dst_float = _mm256_cvtph_ps(dst_half);

        __m256 sum = _mm256_add_ps(dst_float, a);

        // Convert back to half precision
        __m128i sum_half = _mm256_cvtps_ph(sum, 0);

        _mm_store_si128((__m128i *)&dst[ii], sum_half);
    }

    // Handle remaining elements with simple scalar addition
    int remaining_start = num_avx_chunks * AVX_SIZE;
    for (int i = remaining_start; i < mat_size; i++)
    {
        // Assuming a software function to add half floats
        dst[i] = dst[i] = float16_from_float(A_uint16[i] + dst_uint16[i]);
    }
}
/*  ------------------------------------------------------------------------------*/
/*  vec1_avx_add_float32 : Add two 1D float32 vector using AVX intel_intrinsics.        */
/*  ------------------------------------------------------------------------------*/
void vec1_avx_add_float32(float32 *dst, float32 *A, int mat_size)
{
    int AVX_SIZE = 8; // AVX can process 8 floats at a time
    int num_avx_chunks = mat_size / AVX_SIZE;

    for (uint32_t i = 0; i < num_avx_chunks; i++)
    {
        // compute the starting index for the current chunk
        int ii = i * AVX_SIZE;

        __m256 a = _mm256_load_ps(&A[ii]);
        __m256 dst_chunk = _mm256_load_ps(&dst[ii]);
        __m256 sum = _mm256_add_ps(dst_chunk, a);

        _mm256_store_ps(&dst[ii], sum);
    }

    // Handle remaining elements with simple scalar addition
    int remaining_start = num_avx_chunks * AVX_SIZE;
    for (int i = remaining_start; i < mat_size; i++)
    {
        dst[i] += A[i];
    }
}
/*  ------------------------------------------------------------------------------*/
/*  vec1_avx_add_float64 : Add two 1D float64 vector using AVX intel_intrinsics.        */
/*  ------------------------------------------------------------------------------*/
void vec1_avx_add_float64(float64 *dst, float64 *A, int mat_size)
{
    int AVX_SIZE = 4; // AVX can process 4 double at a time
    int num_avx_chunks = mat_size / AVX_SIZE;

    for (uint32_t i = 0; i < num_avx_chunks; i++)
    {
        // compute the starting index for the current chunk
        int ii = i * AVX_SIZE;

        __m256d a = _mm256_load_pd(&A[ii]);
        __m256d dst_chunk = _mm256_load_pd(&dst[ii]);
        __m256d sum = _mm256_add_pd(dst_chunk, a);

        _mm256_store_pd(&dst[ii], sum);
    }

    // Handle remaining elements with simple scalar addition
    int remaining_start = num_avx_chunks * AVX_SIZE;
    for (int i = remaining_start; i < mat_size; i++)
    {
        dst[i] += A[i];
    }
}

/*  ----------------------------------------------------------------------------*/
/*                                  SSE                                         */
/*  ----------------------------------------------------------------------------*/
/*  ----------------------------------------------------------------------------*/
/*  vec1_sse_mul_float16 : Multiply two 1D float16 vector using AVX intel_intrinsics. */
/*  ----------------------------------------------------------------------------*/
void vec1_sse_mul_float16(float16 *dst, float16 *A, float16 *B, int mat_size)
{
    int SSE_SIZE = 4; // SSE2 can process 4 half floats at a time
    int num_sse_chunks = mat_size / SSE_SIZE;

    for (uint32_t i = 0; i < num_sse_chunks; i++)
    {
        // compute the starting index for the current chunk
        int ii = i * SSE_SIZE;

        // Load data
        __m128i a_half = _mm_load_si128((__m128i *)&A[ii]);
        __m128i b_half = _mm_load_si128((__m128i *)&B[ii]);

        // Convert to single precision
        __m128 a = cvtph_ps(a_half);
        __m128 b = cvtph_ps(b_half);

        // Perform multiplication
        __m128 product = _mm_mul_ps(a, b);

        // Convert back to half precision
        __m128i product_half = cvtps_ph(product);

        // Store result
        _mm_store_si128((__m128i *)&dst[ii], product_half);
    }

    // Handle remaining elements with simple scalar multiplication
    int remaining_start = num_sse_chunks * SSE_SIZE;
    for (int i = remaining_start; i < mat_size; i++)
    {
        dst[i] = float16_mult(A[i], B[i]);
    }
}

// Function to convert half precision floats to single precision
__m128 cvtph_ps(__m128i a)
{
    // Constants for the conversion
    const __m128i expMask = _mm_set1_epi32(0x7C00);
    const __m128i expAdjust = _mm_set1_epi32(0x1C000);
    const __m128i infNanExp = _mm_set1_epi32(0x7F800000);

    // Extract the sign, exponent, and significand
    __m128i sign = _mm_and_si128(_mm_slli_epi32(a, 16), _mm_set1_epi32(0x80000000));
    __m128i exp = _mm_and_si128(_mm_slli_epi32(a, 13), infNanExp);
    __m128i mantissa = _mm_slli_epi32(a, 13);

    // Adjust the exponent
    exp = _mm_add_epi32(exp, expAdjust);

    // Combine the sign, exponent, and significand
    __m128i result = _mm_or_si128(sign, _mm_or_si128(exp, mantissa));

    // Cast the result to single precision
    return _mm_castsi128_ps(result);
}

// Function to convert single precision floats to half precision
__m128i cvtps_ph(__m128 a)
{
    // Constants for the conversion
    const __m128i signMask = _mm_set1_epi32(0x80000000);
    const __m128i expMask = _mm_set1_epi32(0x7F800000);
    const __m128i mantissaMask = _mm_set1_epi32(0x007FFFFF);
    const __m128i expAdjust = _mm_set1_epi32((127 - 15) << 23);

    // Cast the input to integer
    __m128i ai = _mm_castps_si128(a);

    // Extract the sign, exponent, and significand
    __m128i sign = _mm_and_si128(ai, signMask);
    __m128i exp = _mm_and_si128(ai, expMask);
    __m128i mantissa = _mm_and_si128(ai, mantissaMask);

    // Adjust the exponent
    exp = _mm_sub_epi32(exp, expAdjust);

    // Combine the sign, exponent, and significand, and shift to half precision
    __m128i result = _mm_or_si128(sign, _mm_or_si128(exp, mantissa));
    result = _mm_srli_epi32(result, 13);

    // Pack the result to 16-bit
    return _mm_packs_epi32(result, result);
}
/*  ------------------------------------------------------------------------------*/
/*  vec1_sse_mul_float32 : Multiply two 1D float32 vector using AVX intel_intrinsics.   */
/*  ------------------------------------------------------------------------------*/
void vec1_sse_mul_float32(float32 *dst, float32 *A, float32 *B, int mat_size)
{
    int SSE_SIZE = 4; // SSE2 can process 4 floats at a time
    int num_sse_chunks = mat_size / SSE_SIZE;

    for (uint32_t i = 0; i < num_sse_chunks; i++)
    {
        // compute the starting index for the current chunk
        int ii = i * SSE_SIZE;

        // Load data
        __m128 a = _mm_load_ps(&A[ii]);
        __m128 b = _mm_load_ps(&B[ii]);

        // Perform multiplication
        __m128 product = _mm_mul_ps(a, b);

        // Store result
        _mm_store_ps(&dst[ii], product);
    }

    // Handle remaining elements with simple scalar multiplication
    int remaining_start = num_sse_chunks * SSE_SIZE;
    for (int i = remaining_start; i < mat_size; i++)
    {
        dst[i] = A[i] * B[i];
    }
}
/*  ------------------------------------------------------------------------------*/
/*  vec1_sse_mul_float32 : Multiply two 1D float64 vector using AVX intel_intrinsics.   */
/*  ------------------------------------------------------------------------------*/
void vec1_sse_mul_float64(float64 *dst, float64 *A, float64 *B, int mat_size)
{
    int SSE_SIZE = 2; // SSE2 can process 2 doubles at a time
    int num_sse_chunks = mat_size / SSE_SIZE;

    for (uint32_t i = 0; i < num_sse_chunks; i++)
    {
        // compute the starting index for the current chunk
        int ii = i * SSE_SIZE;

        // Load data
        __m128d a = _mm_load_pd(&A[ii]);
        __m128d b = _mm_load_pd(&B[ii]);

        // Perform multiplication
        __m128d product = _mm_mul_pd(a, b);

        // Store result
        _mm_store_pd(&dst[ii], product);
    }

    // Handle remaining elements with simple scalar multiplication
    int remaining_start = num_sse_chunks * SSE_SIZE;
    for (int i = remaining_start; i < mat_size; i++)
    {
        dst[i] = A[i] * B[i];
    }
}
/*  ------------------------------------------------------------------------------*/
/*  vec1_sse_add_float16 : Add two 1D float16 vector using AVX intel_intrinsics.        */
/*  ------------------------------------------------------------------------------*/
void vec1_sse_add_float16(float16 *dst, float16 *A, int mat_size)
{
    int AVX_SIZE = 8; // F16C process 8 half floats at a time
    int num_sse_chunks = mat_size / AVX_SIZE;

    for (uint32_t i = 0; i < num_sse_chunks; i++)
    {
        // compute the starting index for the current chunk
        int ii = i * AVX_SIZE;

        // Load data
        __m128i a_half = _mm_load_si128((__m128i *)&A[ii]);
        __m128i dst_half = _mm_load_si128((__m128i *)&dst[ii]);

        // Convert to single precision
        __m128 a = cvtph_ps(a_half);
        __m128 dst_float = cvtph_ps(dst_half);

        __m128 sum = _mm_add_ps(dst_float, a);

        // Convert back to half precision
        __m128i sum_half = cvtps_ph(sum);

        _mm_store_si128((__m128i *)&dst[ii], sum_half);
    }

    // Handle remaining elements with simple scalar addition
    int remaining_start = num_sse_chunks * AVX_SIZE;
    for (int i = remaining_start; i < mat_size; i++)
    {
        // Assuming a software function to add half floats
        dst[i] = float16_add(A[i], dst[i]);
    }
}
/*  ------------------------------------------------------------------------------*/
/*  vec1_sse_add_float32 : Add two 1D float32 vector using AVX intel_intrinsics.        */
/*  ------------------------------------------------------------------------------*/
void vec1_sse_add_float32(float32 *dst, float32 *A, int mat_size)
{
    int SSE_SIZE = 4; // SSE2 can process 4 floats at a time
    int num_sse_chunks = mat_size / SSE_SIZE;

    for (uint32_t i = 0; i < num_sse_chunks; i++)
    {
        int ii = i * SSE_SIZE;

        __m128 a = _mm_load_ps(&A[ii]);
        __m128 dst_chunk = _mm_load_ps(&dst[ii]);
        __m128 sum = _mm_add_ps(dst_chunk, a);

        _mm_store_ps(&dst[ii], sum);
    }

    int remaining_start = num_sse_chunks * SSE_SIZE;
    for (int i = remaining_start; i < mat_size; i++)
    {
        dst[i] += A[i];
    }
}
/*  ------------------------------------------------------------------------------*/
/*  vec1_sse_add_float64 : Add two 1D float64 vector using AVX intel_intrinsics.        */
/*  ------------------------------------------------------------------------------*/
void vec1_sse_add_float64(float64 *dst, float64 *A, int mat_size)
{
    int SSE_SIZE = 2; // SSE2 can process 2 doubles at a time
    int num_sse_chunks = mat_size / SSE_SIZE;

    for (uint32_t i = 0; i < num_sse_chunks; i++)
    {
        int ii = i * SSE_SIZE;

        __m128d a = _mm_load_pd(&A[ii]);
        __m128d dst_chunk = _mm_load_pd(&dst[ii]);
        __m128d sum = _mm_add_pd(dst_chunk, a);

        _mm_store_pd(&dst[ii], sum);
    }

    int remaining_start = num_sse_chunks * SSE_SIZE;
    for (int i = remaining_start; i < mat_size; i++)
    {
        dst[i] += A[i];
    }
}