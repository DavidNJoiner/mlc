#ifndef DTYPE_H_ 
#define DTYPE_H_

#include <stdlib.h>
#include <time.h> 

// Dtypes
typedef float float32;
typedef double float64;

// The stride is contained within the dtype
#define FLOAT32 sizeof(float32)
#define FLOAT64 sizeof(float64)

// If using CUDA, include the CUDA runtime
#ifdef __CUDACC__
    #include <cuda_runtime.h>
    #include "cuda.h"
#endif
    typedef __half float16;
    #define FLOAT16 sizeof(float16)

//  Data object struct
typedef struct {
    void* values;
    int size;
    int dim;
    int* shape;
    int dtype;
} Data;

// Prototypes
void            flattenArray(void* array, void* flattened, int* shape, int dim, int dtype, int idx);
void            printData(Data* dat);
int             GetDtypeSize(int dtype);
int             calculateIndex(int* indices, int* strides, int dim);
const char*     GetDType(int num);
Data*           convertToData(void* array, int* shape, int dim, int dtype);
Data*           randomData(int size, int* range, int* shape, int dim, int dtype);

#endif //DTYPE_H

#ifndef DTYPE_IMPLEMENTATION
#define DTYPE_IMPLEMENTATION
/*  -------------------------------------------------------*/
/*  dtypes functions                                       */
/*  -------------------------------------------------------*/
const char* GetDType(int dtype) {
    switch(dtype) {
        case FLOAT32: return "float32";
        case FLOAT64: return "float64";
        case FLOAT16: return "float16";
        default: return "Unknown dtype";
    }
}

int GetDtypeSize(int dtype) {
    switch (dtype) {
        case FLOAT32: return sizeof(float32);
        case FLOAT64: return sizeof(float64);
        case FLOAT16: return sizeof(float16);
        default: return 0;
    }
}
/*  -----------------------------------------------------------------------*/
/*  calculateIndex : convert multi-dimensional index into a linear index;  */
/*  -----------------------------------------------------------------------*/
int calculateIndex(int* indices, int* shape, int dim) {
    int index = 0;
    int stride = 1;
    for (int i = dim - 1; i >= 0; i--) {
        index += indices[i] * stride;
        stride *= shape[i];
    }
    return index;
}
/*  ---------------------------------------------------------------------------------------------*/
/*  flattenArray : recursively flattens a multi-dimensional array into a one-dimensional array.  */
/*  ---------------------------------------------------------------------------------------------*/
void flattenArray(void* array, void* flattened, int* shape, int dim, int dtype, int idx) {
    if (dim == 0) {
        switch (dtype) {
            case FLOAT16: {
                float16* farray = (float16*)array;
                float16* fflattened = (float16*)flattened;
                fflattened[idx] = *farray;
                break;
            }
            case FLOAT32: {
                float32* farray = (float32*)array;
                float32* fflattened = (float32*)flattened;
                fflattened[idx] = *farray;
                break;
            }
            case FLOAT64: {
                float64* farray = (float64*)array;
                float64* fflattened = (float64*)flattened;
                fflattened[idx] = *farray;
                break;
            }
            default:
                printf("Failed to flatten : Unsupported dtype %s\n", GetDType(dtype));
                break;
        }
    } else {
        int stride = 1;
        for (int i = 1; i < dim; i++) {
            stride *= shape[i];
        }
        stride *= dtype;  // calculate the stride for the current dimension
        for (int i = 0; i < shape[0]; i++) {
            flattenArray((char*)array + i * stride, flattened, shape + 1, dim - 1, dtype, idx + i * stride / dtype);
        }
    }
}
/*  --------------------------------------------------------------------------------*/
/*  convertToData : Converts a given multi-dimensional array into a Data structure. */
/*  --------------------------------------------------------------------------------*/
Data* convertToData(void* array, int* shape, int dim, int dtype) {
    int size = 1;
    for (int i = 0; i < dim; i++) {
        size *= shape[i];
    }
    
    int byte_size =  size * GetDtypeSize(dtype);
    void* flattened = (float32 *)aligned_alloc(32, byte_size);

    flattenArray(array, flattened, shape, dim, dtype, 0);

    Data* data = (Data*)malloc(sizeof(Data));
    data->values = flattened;
    data->size = size;
    data->dim = dim;
    data->shape = shape;
    data->dtype = dtype;

    return data;
}
/*  -----------------------------------------------------------------------------*/
/*  printData : Converts a given multi-dimensional array into a Data structure.  */
/*  -----------------------------------------------------------------------------*/
void printData(Data* dat) {
    const char* dtypeStr = GetDType(dat->dtype);
    if (dtypeStr == NULL) {
        printf("Error: GetDType returned NULL.\n");
        return;
    }

    printf("dtype : %4s \n", dtypeStr);
    printf("shape : ");
    for (int i = 0; i < dat->dim; i++) {
        printf("%2d", dat->shape[i]);
    }
    printf("\n");
    printf("dimension : %d \n \n", dat->dim);

    if (dat->dim >= 0) {
        print_op(dat, dat->dim);
    } else {
        printf("Error: Invalid dimension for printOp.\n");
    }
}
/*  -----------------------------------------------------------------------------*/
/*  randomData : Generate a Data object filled with random values.               */
/*  -----------------------------------------------------------------------------*/
Data* randomData(int size, int* range, int* shape, int dim, int dtype) {
    // Seed the random number generator
    srand((unsigned int)time(NULL));
    
    int dtypeSize = GetDtypeSize(dtype);
    const char* dtypeName = GetDType(dtype);
    int byte_size = size * dtypeSize;
    int alignment = 32;
    
    // Making sure byte_size is a multiple of the alignment
    if (byte_size % alignment != 0) {
        byte_size = ((byte_size / alignment) + 1) * alignment;
    }
    
    printf("size: %d, dtype: %s, byte_size: %d, alignment: %d\n", size, dtypeName, byte_size, alignment);
    void* random_values = aligned_alloc(alignment, byte_size);

    if(dtype == FLOAT16) {
        float16* ptr = (float16*)random_values;
        for(int i = 0; i < size; i++) {
            ptr[i] = (float16)(range[0] + ((float)rand() / (float)RAND_MAX) * (range[1] - range[0]));
        }
    }
    if(dtype == FLOAT32) {
        float32* ptr = (float32*)random_values;
        for(int i = 0; i < size; i++) {
            ptr[i] = (float32)(range[0] + ((float)rand() / (float)RAND_MAX) * (range[1] - range[0]));
        }
    } else if(dtype == FLOAT64) {
        float64* ptr = (float64*)random_values;
        for(int i = 0; i < size; i++) {
            ptr[i] = (float64)(range[0] + ((double)rand() / (double)RAND_MAX) * (range[1] - range[0]));
        }
    }
    
    Data* data = (Data*)malloc(sizeof(Data));
    data->values = random_values;
    data->size = size;
    data->dim = dim;
    data->shape = shape;
    data->dtype = dtype;
    
    return data;
}
#endif //DTYPE_IMPLEMENTATION