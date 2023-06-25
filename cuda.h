#ifndef CUDA_H_ 
#define CUDA_H_

#include "dtype.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <cuda_runtime.h>

__global__ void vec1_cuda_mul_float32(float32* res, float32* mat1, float32* mat2, int mat_size);
__global__ void vec1_cuda_mul_float64(float64* res, float64* mat1, float64* mat2, int mat_size);
__global__ void vec1_cuda_add_float32(float32* res, float32* mat1, int mat_size);
__global__ void vec1_cuda_add_float64(float64* res, float64* mat1, int mat_size);

#endif //CUDA_H_

#ifndef CUDA_IMPLEMENTATION
#define CUDA_IMPLEMENTATION

/*
    -------------------------------------------------------
    vec1_cuda_mul_float32 : Multiply two 1D float32 vector using CUDA.
    -------------------------------------------------------
*/
__global__ void vec1_cuda_mul_float32(float32* res, float32* mat1, float32* mat2, int mat_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < mat_size) {
        res[idx] = mat1[idx] * mat2[idx];
    }
}

/*
    -------------------------------------------------------
    vec1_cuda_mul_float64 : Multiply two 1D float64 vector using CUDA.
    -------------------------------------------------------
*/
__global__ void vec1_cuda_mul_float64(float64* res, float64* mat1, float64* mat2, int mat_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < mat_size) {
        res[idx] = mat1[idx] * mat2[idx];
    }
}

/*
    -------------------------------------------------------
    vec1_cuda_add_float32 : Add two 1D float32 vector using CUDA.
    -------------------------------------------------------
*/
__global__ void vec1_cuda_add_float32(float32* res, float32* mat1, int mat_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < mat_size) {
        res[idx] += mat1[idx];
    }
}

/*
    -------------------------------------------------------
    vec1_cuda_add_float64 : Add two 1D float64 vector using CUDA.
    -------------------------------------------------------
*/
__global__ void vec1_cuda_add_float64(float64* res, float64* mat1, int mat_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < mat_size) {
        res[idx] += mat1[idx];
    }
}

#endif //CUDA_IMPLEMENTATION
