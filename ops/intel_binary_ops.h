#ifndef INTEL_BINARY_OPS_H
#define INTEL_BINARY_OPS_H

#include <immintrin.h>
#include "../core/config.h"
#include "../core/types/dtype.h"

/*-------------------------------------------------------*/
/*                  Functions pointers                   */
/*-------------------------------------------------------*/
// Function pointer for flexibility over different data types and SIMD instructions (WIP)
typedef void (*MulFuncPtr)(void*, void*, void*, int);
typedef void (*AddFuncPtr)(void*, void*, int);

extern MulFuncPtr mul_1D_f16;
extern MulFuncPtr mul_1D_f32;
extern MulFuncPtr mul_1D_f64;
extern AddFuncPtr add_1D_f16;
extern AddFuncPtr add_1D_f32;
extern AddFuncPtr add_1D_f64;

/*-------------------------------------------------------*/
/*                 Functions prototypes                  */
/*-------------------------------------------------------*/
// float16 Conversions
__m128 cvtph_ps(__m128i a);
__m128i cvtps_ph(__m128 a);

#if defined(AVX512)
// TODO : Operation using appropriate 512 bytes vector types (__m512, __m512d, etc.).
#endif // AVX512

#if defined(AVX)
// Multiplications
void vec1_avx_mul_float16(float16 *dst, float16 *mat1, float16 *mat2, int mat_size);
void vec1_avx_mul_float32(float32 *res, float32 *mat1, float32 *mat2, int mat_size);
void vec1_avx_mul_float64(float64 *res, float64 *mat1, float64 *mat2, int mat_size);
// Additions
void vec1_avx_add_float16(float16 *dst, float16 *mat1, int mat_size);
void vec1_avx_add_float32(float32 *res, float32 *mat1, int mat_size);
void vec1_avx_add_float64(float64 *res, float64 *mat1, int mat_size);

//MulFuncPtr mul_1D_f16 = &vec1_avx_mul_float16;
//MulFuncPtr mul_1D_f32 = &vec1_avx_mul_float32;
//MulFuncPtr mul_1D_f64 = &vec1_avx_mul_float64;
//AddFuncPtr add_1D_f16 = &vec1_avx_add_float16;
//AddFuncPtr add_1D_f32 = &vec1_avx_add_float32;
//AddFuncPtr add_1D_f64 = &vec1_avx_add_float64;
#endif // AVX

#if defined(SSE)
// Multiplications
void vec1_sse_mul_float16(float16 *dst, float16 *mat1, float16 *mat2, int mat_size);
void vec1_sse_mul_float32(float32 *res, float32 *mat1, float32 *mat2, int mat_size);
void vec1_sse_mul_float64(float64 *res, float64 *mat1, float64 *mat2, int mat_size);
// Additions
void vec1_sse_add_float16(float16 *dst, float16 *mat1, int mat_size);
void vec1_sse_add_float32(float32 *res, float32 *mat1, int mat_size);
void vec1_sse_add_float64(float64 *res, float64 *mat1, int mat_size);

MulFuncPtr mul_1D_f16 = &vec1_sse_mul_float16;
MulFuncPtr mul_1D_f32 = &vec1_sse_mul_float32;
MulFuncPtr mul_1D_f64 = &vec1_sse_mul_float64;
AddFuncPtr add_1D_f16 = &vec1_sse_add_float16;
AddFuncPtr add_1D_f32 = &vec1_sse_add_float32;
AddFuncPtr add_1D_f64 = &vec1_sse_add_float64;
#endif // SSE


#endif // INTEL_BINARY_OPS_H