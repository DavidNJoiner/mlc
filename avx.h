#ifndef AVX_H_ 
#define AVX_H_

#include <immintrin.h>
#include "dtype.h"

// Multiplications
void    vec1_avx_mul_float16(float16* dst, float16* mat1, float16* mat2, int mat_size);
void    vec1_avx_mul_float32(float32* res, float32* mat1, float32* mat2, int mat_size);
void    vec1_avx_mul_float64(float64* res, float64* mat1, float64* mat2, int mat_size);
// Additions
void    vec1_avx_add_float16(float16* dst, float16* mat1, int mat_size);
void    vec1_avx_add_float32(float32* res, float32* mat1, int mat_size);
void    vec1_avx_add_float64(float64* res, float64* mat1, int mat_size);

#endif //AVX_H