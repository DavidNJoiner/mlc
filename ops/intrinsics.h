#ifndef INTRINSICS_H_
#define INTRINSICS_H_

#include <immintrin.h>
#include "../core/config.h"
#include "../core/types/dtype.h"

#if defined(AVX512)
// TODO : Operation using appropriate 512 bytes vector types (__m512, __m512d, etc.).

#elif defined(AVX2) || defined(AVX)
// Multiplications
void vec1_avx_mul_float16(float16 *dst, float16 *mat1, float16 *mat2, int mat_size);
void vec1_avx_mul_float32(float32 *res, float32 *mat1, float32 *mat2, int mat_size);
void vec1_avx_mul_float64(float64 *res, float64 *mat1, float64 *mat2, int mat_size);
// Additions
void vec1_avx_add_float16(float16 *dst, float16 *mat1, int mat_size);
void vec1_avx_add_float32(float32 *res, float32 *mat1, int mat_size);
void vec1_avx_add_float64(float64 *res, float64 *mat1, int mat_size);

#elif defined(__SSE2__)
// Conversions
__m128 cvtph_ps(__m128i a);
__m128i cvtps_ph(__m128 a);
// Multiplications
void vec1_sse_mul_float16(float16 *dst, float16 *mat1, float16 *mat2, int mat_size);
void vec1_sse_mul_float32(float32 *res, float32 *mat1, float32 *mat2, int mat_size);
void vec1_sse_mul_float64(float64 *res, float64 *mat1, float64 *mat2, int mat_size);
// Additions
void vec1_sse_add_float16(float16 *dst, float16 *mat1, int mat_size);
void vec1_sse_add_float32(float32 *res, float32 *mat1, int mat_size);
void vec1_sse_add_float64(float64 *res, float64 *mat1, int mat_size);
#endif

#endif // INTRINSICS_H_