#ifndef DTYPE_H_ 
#define DTYPE_H_

#ifdef __CUDACC__

#include <cuda_fp16.h>
typedef __half float16;
#else
typedef unsigned short float16;
#endif

typedef float float32;
typedef double float64;

#define FLOAT16 sizeof(float16)
#define FLOAT32 sizeof(float32)
#define FLOAT64 sizeof(float64)



#endif // DTYPE_H
