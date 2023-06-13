#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dtype.h"

#ifndef TENSOR_H_ 
#define TENSOR_H_

typedef struct {
    Data* data;
    float32* gradient;
    int* shape;
    int dim;
    int stride;
} Tensor;

Tensor* createTensor(Data* data);
void mult(Tensor* dst, Tensor* A, Tensor* B);
void add(Tensor* dst, Tensor* A);
void printTensor(Tensor* A);

#endif //TENSOR_H

#ifndef TENSOR_IMPLEMENTATION 
#define TENSOR_IMPLEMENTATION

Tensor* createTensor(Data* data) {
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));

    tensor->data = data;
    tensor->gradient = (float32*)calloc(data->size, sizeof(float32));
    tensor->shape = data->shape;
    tensor->dim = 2;  // Assuming 2 dimensions for simplicity
    tensor->stride = data->stride;

    return tensor;
}

void mult(Tensor* dst, Tensor* A, Tensor* B) {
    // Assuming A, B and dst are all of the same shape and dimension
    for (int i = 0; i < A->shape[0]; ++i) {
        for (int j = 0; j < A->shape[1]; ++j) {
            int index = i * A->shape[1] + j;
            dst->data->values[index] = A->data->values[index] * B->data->values[index];
        }
    }
}

void add(Tensor* dst, Tensor* A) {
    for (int i = 0; i < A->shape[0]; ++i) {
        for (int j = 0; j < A->shape[1]; ++j) {
            int index = i * A->shape[1] + j;
            dst->data->values[index] += A->data->values[index];
        }
    }
}

void printTensor(Tensor* A) {
    for (int i = 0; i < A->shape[0]; ++i) {
        for (int j = 0; j < A->shape[1]; ++j) {
            int index = i * A->shape[1] + j;
            printf("%.2f ", A->data->values[index]);
        }
        printf("\n");
    }
}

#endif //TENSOR_IMPLEMENTATION