#ifndef DTYPE_H_ 
#define DTYPE_H_

typedef float float32;
typedef double float64;
typedef unsigned short float16;
typedef unsigned short bfloat16;
typedef struct { float real; float imag; } complex32;
typedef struct { double real; double imag; } complex64;
typedef struct { long double real; long double imag; } complex128;
typedef unsigned char uint8;
typedef signed char int8;
typedef short int16;
typedef int int32;
typedef long long int int64;
typedef unsigned char quint8;
typedef signed char qint8;
typedef int qint32;
typedef signed char quint4x2;


//The stride is contained within the dtype
#define FLOAT32 sizeof(float32)
#define FLOAT64 sizeof(float64)
#define FLOAT16 sizeof(float16)
#define BFLOAT16 sizeof(bfloat16)
#define COMPLEX32 sizeof(complex32)
#define COMPLEX64 sizeof(complex64)
#define COMPLEX128 sizeof(complex128)
#define UINT8 sizeof(uint8)
#define INT8 sizeof(int8)
#define INT16 sizeof(int16)
#define INT32 sizeof(int32)
#define INT64 sizeof(int64)
#define QUINT8 sizeof(quint8)
#define QINT8 sizeof(qint8)
#define QINT32 sizeof(qint32)
#define QUINT4X2 sizeof(quint4x2)


//Data object struct
typedef struct {
    void* values;
    int size;
    int dim;
    int* shape;
    int dtype;
} Data;

const char* GetDType(int num);
int dtypeSize(int dtype);
int calculateIndex(int* indices, int* strides, int dim);
void flattenArray(void* array, void* flattened, int* shape, int dim, int dtype, int idx);
Data* convertToData(void* array, int* shape, int dim, int dtype);


#endif //DTYPE_H

#ifndef DTYPE_IMPLEMENTATION
#define DTYPE_IMPLEMENTATION


const char* GetDType(int dtype) {
    switch(dtype) {
        case FLOAT32: return "float32";
        case FLOAT64: return "float64";
        case FLOAT16: return "float16";
        default: return "Unknown dtype";
    }
}

int dtypeSize(int dtype) {
    switch (dtype) {
        case FLOAT32: return sizeof(float32);
        case FLOAT64: return sizeof(float64);
        case FLOAT16: return sizeof(float16);
        // add other cases as needed
        default: return 0;
    }
}

/*
    -------------------------------------------------------
    calculateIndex : convert multi-dimensional index into a linear index
    -------------------------------------------------------
*/
int calculateIndex(int* indices, int* shape, int dim) {
    int index = 0;
    int stride = 1;
    for (int i = dim - 1; i >= 0; i--) {
        index += indices[i] * stride;
        stride *= shape[i];
    }
    return index;
}

void flattenArray(void* array, void* flattened, int* shape, int dim, int dtype, int idx) {
    if (dim == 1) {
        int elementSize = dtype;  // use dtype as element size
        switch (dtype) {
            case FLOAT32: {
                float32* farray = (float32*)array;
                float32* fflattened = (float32*)flattened;
                for (int i = 0; i < shape[0]; i++) {
                    fflattened[i + idx] = farray[i];
                }
                break;
            }
            case FLOAT64: {
                float64* farray = (float64*)array;
                float64* fflattened = (float64*)flattened;
                for (int i = 0; i < shape[0]; i++) {
                    fflattened[i + idx] = farray[i];
                }
                break;
            }
            default:
                printf("Unsupported dtype %d\n", dtype);
                break;
        }
    } else {
        int stride = shape[1] * dtype;  // calculate the stride for the current dimension
        for (int i = 0; i < shape[0]; i++) {
            flattenArray((char*)array + i * stride, flattened, shape + 1, dim - 1, dtype, idx + i * shape[1]);
        }
    }
}

Data* convertToData(void* array, int* shape, int dim, int dtype) {
    int size = 1;
    for (int i = 0; i < dim; i++) {
        size *= shape[i];
    }
    
    void* flattened = (float32 *)aligned_alloc(32, size * dtypeSize(dtype));

    flattenArray(array, flattened, shape, dim, dtype, 0);

    Data* data = (Data*)malloc(sizeof(Data));
    data->values = flattened;
    data->size = size;
    data->dim = dim;
    data->shape = shape;
    data->dtype = dtype;

    return data;
}

#endif //DTYPE_IMPLEMENTATION