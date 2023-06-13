#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    float* values;
    int size;
} Data;

typedef struct {
    Data* data;
    float* gradient;
    int* shape;
    int dim;
    int stride;
} Tensor;

Data* makedat(float* arr, size_t size) {                                                                                                                                            
    Data* dat = (Data*)malloc(sizeof(Data));
    dat->size = (int)size; 
    dat->values = (float*) malloc(dat->size * sizeof(float));

    // Copy values from input array to Data struct
    for(int i = 0; i < size; i++) {
        dat->values[i] = arr[i];
    }

    return dat;                                                                                                                                                                 
}

Tensor* createTensor(Data* data, int rows, int cols, int dim) {
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    tensor->data = malloc(sizeof(*(tensor->data)) * rows * cols );
    tensor->gradient = (float*)calloc(rows * cols, sizeof(float));
    tensor->shape = (int*)malloc(sizeof(int) * dim);	
    tensor->stride = sizeof(sizeof(float)*cols); //Assuming the datatype of the dataset is float
    tensor->shape[0] = rows;
    tensor->shape[1] = cols;
    tensor->dim = dim;
    tensor->data = data;
    return tensor;
}

int main() {

    float arr[4][3] = {{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}};
    printf("%zu", sizeof(arr)/sizeof(int));
    return 0;
}

