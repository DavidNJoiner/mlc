#include "cuda_ops.h"
#include <stdio.h>

#ifdef __CUDACC__
    #include <cuda_runtime.h>
    #define DEEPC_CUDA true
    /*  ---------------------------------------------------------------------------------*/
    /*  vec1_kernel_mul_float16 : Multiply two 1D float16 vector using CUDA Instructions.  */
    /*  ---------------------------------------------------------------------------------*/
    __global__ void vec1_kernel_mul_float16(float16* dst, float16* A, float16* B, int mat_size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < mat_size) {
            dst[i] = __hmul(A[i], B[i]);
        }
    }
    /*  -----------------------------------------------------------------------------------*/
    /*  vec1_kernel_mul_float32 : Multiply two 1D float32 vector using CUDA Instructions.    */
    /*  -----------------------------------------------------------------------------------*/
    __global__ void vec1_kernel_mul_float32(float32* dst, float32* A, float32* B, int mat_size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < mat_size) {
            dst[i] = A[i] * B[i];
        }
    }
    /*  -----------------------------------------------------------------------------------*/
    /*  vec1_kernel_mul_float64 : Multiply two 1D float64 vector using CUDA Instructions.    */
    /*  -----------------------------------------------------------------------------------*/
    __global__ void vec1_kernel_mul_float64(float64* dst, float64* A, float64* B, int mat_size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < mat_size) {
            dst[i] = A[i] * B[i];
        }
    }
    /*  -----------------------------------------------------------------------------------*/
    /*  vec1_kernel_add_float16 : Add two 1D float16 vector using CUDA Instructions.         */
    /*  -----------------------------------------------------------------------------------*/
    __global__ void vec1_kernel_add_float16(float16* dst, float16* A, int mat_size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < mat_size) {
            dst[i] = __hadd(dst[i], A[i]);
        }
    }
    /*  -----------------------------------------------------------------------------------*/
    /*  vec1_kernel_add_float32 : Add two 1D float32 vector using CUDA Instructions.         */
    /*  -----------------------------------------------------------------------------------*/
    __global__ void vec1_kernel_add_float32(float32* dst, float32* A, int mat_size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < mat_size) {
            dst[i] = dst[i] + A[i];
        }
    }
    /*  -----------------------------------------------------------------------------------*/
    /*  vec1_cuda_add_float64 : Add two 1D float64 vector using CUDA Instructions.         */
    /*  -----------------------------------------------------------------------------------*/
    __global__ void vec1_kernel_add_float64(float64* dst, float64* A, int mat_size)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < mat_size) {
            dst[i] = dst[i] + A[i];
        }
    }
    #ifdef __cplusplus
    extern "C" {
    #endif
        void vec1_cuda_mul_float16(float16* dst, float16* A, float16* B, int mat_size) {
            int threads = 256;
            int blocks = (mat_size + threads - 1) / threads;
            vec1_kernel_mul_float16<<<blocks, threads>>>(dst, A, B, mat_size);
            //cudaDeviceSynchronize();
        }

        void vec1_cuda_mul_float32(float32* dst, float32* A, float32* B, int mat_size) {
            int threads = 256;
            int blocks = (mat_size + threads - 1) / threads;
            vec1_kernel_mul_float32<<<blocks, threads>>>(dst, A, B, mat_size);
            //cudaDeviceSynchronize();
        }

        void vec1_cuda_mul_float64(float64* dst, float64* A, float64* B, int mat_size) {
            int threads = 256;
            int blocks = (mat_size + threads - 1) / threads;
            vec1_kernel_mul_float64<<<blocks, threads>>>(dst, A, B, mat_size);
            //cudaDeviceSynchronize();
        }

        void vec1_cuda_add_float16(float16* dst, float16* A, int mat_size) {
            int threads = 256;
            int blocks = (mat_size + threads - 1) / threads;
            vec1_kernel_add_float16<<<blocks, threads>>>(dst, A, mat_size);
            //cudaDeviceSynchronize();
        }

        void vec1_cuda_add_float32(float32* dst, float32* A, int mat_size) {
            int threads = 256;
            int blocks = (mat_size + threads - 1) / threads;
            vec1_kernel_add_float32<<<blocks, threads>>>(dst, A, mat_size);
            //cudaDeviceSynchronize();
        }

        void vec1_cuda_add_float64(float64* dst, float64* A, int mat_size) {
            int threads = 256;
            int blocks = (mat_size + threads - 1) / threads;
            vec1_kernel_add_float64<<<blocks, threads>>>(dst, A, mat_size);
            //cudaDeviceSynchronize();
        }
    #ifdef __cplusplus
    }
    #endif
#else
    #define DEEPC_CUDA false  
#endif




