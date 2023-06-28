#ifndef CUDA_H_ 
#define CUDA_H_

typedef __half float16;
#define FLOAT16 sizeof(float16)
#include "config.h"

__global__ void cuda_mul_float16(float16* dst, float16* A, float16* B, int mat_size);
__global__ void vec1_cuda_mul_float32(float32* res, float32* mat1, float32* mat2, int mat_size);
__global__ void vec1_cuda_mul_float64(float64* res, float64* mat1, float64* mat2, int mat_size);
__global__ void cuda_add_float16(float16* dst, float16* A, int mat_size);
__global__ void vec1_cuda_add_float32(float32* res, float32* mat1, int mat_size);
__global__ void vec1_cuda_add_float64(float64* res, float64* mat1, int mat_size);

#endif //CUDA_H_

#ifndef CUDA_IMPLEMENTATION
#define CUDA_IMPLEMENTATION
/*  ----------------------------------------------------------------------------*/
/*  cuda_mul_float16 : Multiply two 1D float16 vector using CUDA Instructions.  */
/*  ----------------------------------------------------------------------------*/
__global__ void cuda_mul_float16(float16* dst, float16* A, float16* B, int mat_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < mat_size) {
        dst[i] = __hmul2(A[i], B[i]);
    }
}
/*  ------------------------------------------------------------------------------*/
/*  cuda_mul_float32 : Multiply two 1D float32 vector using CUDA Instructions.    */
/*  ------------------------------------------------------------------------------*/
__global__ void cuda_mul_float32(float32* dst, float32* A, float32* B, int mat_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < mat_size) {
        dst[i] = A[i] * B[i];
    }
}
/*  ------------------------------------------------------------------------------*/
/*  cuda_mul_float64 : Multiply two 1D float64 vector using CUDA Instructions.    */
/*  ------------------------------------------------------------------------------*/
__global__ void cuda_mul_float64(float64* dst, float64* A, float64* B, int mat_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < mat_size) {
        dst[i] = A[i] * B[i];
    }
}
/*  ------------------------------------------------------------------------------*/
/*  cuda_add_float16 : Add two 1D float16 vector using CUDA Instructions.         */
/*  ------------------------------------------------------------------------------*/
__global__ void cuda_add_float16(float16* dst, float16* A, int mat_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < mat_size) {
        dst[i] = __hadd(dst[i], A[i]);
    }
}
/*  ------------------------------------------------------------------------------*/
/*  cuda_add_float32 : Add two 1D float32 vector using CUDA Instructions.         */
/*  ------------------------------------------------------------------------------*/
__global__ void cuda_add_float32(float32* dst, float32* A, int mat_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < mat_size) {
        dst[i] = dst[i] + A[i];
    }
}
/*  ------------------------------------------------------------------------------*/
/*  cuda_add_float64 : Add two 1D float64 vector using CUDA Instructions.         */
/*  ------------------------------------------------------------------------------*/
__global__ void cuda_add_float64(float64* dst, float64* A, int mat_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < mat_size) {
        dst[i] = dst[i] + A[i];
    }
}


#endif //CUDA_IMPLEMENTATION
