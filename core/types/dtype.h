#ifndef DTYPE_H_
#define DTYPE_H_
#include <assert.h>

typedef char static_assertion[sizeof(float) == 4 ? 1 : -1];
typedef char static_assertion[sizeof(double) == 8 ? 1 : -1];

// CUDA HALF TYPE
#ifdef __CUDACC__
#include <cuda_fp16.h>
typedef __half float16;
#else
#include "float16.h"
#endif

typedef float float32;
typedef double float64;

#define FLOAT16 sizeof(float16)
#define FLOAT32 sizeof(float32)
#define FLOAT64 sizeof(float64)

const char *get_data_type(int dtype);
int get_data_size(int dtype);

#endif // DTYPE_H
