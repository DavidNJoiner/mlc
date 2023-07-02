#ifndef CUDA_OPS_H_ 
#define CUDA_OPS_H_

#include "dtype.h"

#ifdef __cplusplus
extern "C" {
#endif
void vec1_cuda_mul_float16(float16* dst, float16* A, float16* B, int mat_size);
void vec1_cuda_mul_float32(float32* dst, float32* A, float32* B, int mat_size);
void vec1_cuda_mul_float64(float64* dst, float64* A, float64* B, int mat_size);

void vec1_cuda_add_float16(float16* dst, float16* A, int mat_size);
void vec1_cuda_add_float32(float32* dst, float32* A, int mat_size);
void vec1_cuda_add_float64(float64* dst, float64* A, int mat_size);
#ifdef __cplusplus
}
#endif


#define deepc_cuda true // debug


#endif //CUDA_OPS_H_


