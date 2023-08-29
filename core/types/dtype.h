#ifndef DTYPE_H_
#define DTYPE_H_
#include <assert.h>

static_assert(sizeof(float) == 4, "float is not 32 bits");
static_assert(sizeof(double) == 8, "double is not 64 bits");

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
