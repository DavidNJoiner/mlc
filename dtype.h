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

#define FLOAT32 1
#define FLOAT64 2
#define FLOAT16 3
#define BFLOAT16 4
#define COMPLEX32 5
#define COMPLEX64 6
#define COMPLEX128 7
#define UINT8 8
#define INT8 9
#define INT16 10
#define INT32 11
#define INT64 12
#define QUINT8 13
#define QINT8 14
#define QINT32 15
#define QUINT4X2 16

typedef struct {
    void* values;
    int size;
    int stride;
    int* shape;
    int dtype;
} Data;

typedef struct {
    float* values;
    int size;
    int stride;
    int* shape;
} _Data;

int getDtype(void* array);
Data* convertToData(float32* array, int* shape, int dim);

#endif //DTYPE_H

#ifndef DTYPE_IMPLEMENTATION
#define DTYPE_IMPLEMENTATION


int getDtype(void* array) {
    void;
}

Data* convertToData(float32* array, int* shape, int dim) {
    int size = 1;
    for (int i = 0; i < dim; i++) {
        size *= shape[i];
    }

    float32* flattened = (float32*)malloc(size * sizeof(float32));
    flattenArray(array, flattened, shape, dim);

    Data* data = (Data*)malloc(sizeof(Data));
    data->values = flattened;
    data->size = size;
    data->stride = 1;  // Assuming stride of 1 for flattened array
    data->shape = shape;

    return data;
}

#endif //DTYPE_IMPLEMENTATION