#ifndef CUDA_H_ 
#define CUDA_H_

#include <cuda_runtime.h>
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

#endif //CUDA_H_
