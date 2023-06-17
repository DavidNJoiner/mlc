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


//Data object struct
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

const char* GetDType(int num);
int calculateIndex(int* indices, int* strides, int dim);
void flattenArray(float32* array, float32* flattened, int* shape, int dim);
Data* convertToData(float32* array, int* shape, int dim);


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

int calculateIndex(int* indices, int* strides, int dim) {
    int index = 0;
    for (int i = 0; i < dim; i++) {
        index += indices[i] * strides[i];
    }
    return index;
}

void flattenArray(float32* array, float32* flattened, int* shape, int dim){
    int* indices = (int*)malloc(dim * sizeof(int));
    int* strides = (int*)malloc(dim * sizeof(int));

    // Calculate strides
    strides[dim - 1] = 1;
    for (int i = dim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    // Flatten array
    int flatIndex = 0;
    for (int i = 0; i < dim; i++) {
        indices[i] = 0;
    }

    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++) {
            // Access array using indices and strides
            flattened[flatIndex] = array[calculateIndex(indices, strides, dim)];

            // Update indices
            for (int k = dim - 1; k >= 0; k--) {
                indices[k]++;
                if (indices[k] >= shape[k] && k > 0) {
                    indices[k] = 0;
                } else {
                    break;
                }
            }

            flatIndex++;
        }
    }

    free(indices);
    free(strides);
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
    data->dtype = 1;

    return data;
}
#endif //DTYPE_IMPLEMENTATION