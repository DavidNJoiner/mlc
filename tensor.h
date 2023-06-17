#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dtype.h"
#include "ops.h"

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
    Tensor* new_tensor = (Tensor*)malloc(sizeof(Tensor));

    new_tensor->data = data;
    new_tensor->gradient = (float32*)calloc(data->size, sizeof(float32));
    new_tensor->shape = data->shape;
    new_tensor->dim = 2;  // Assuming 2 dimensions for simplicity
    new_tensor->stride = data->stride;

    return new_tensor;
}

void mult(Tensor* dst, Tensor* A, Tensor* B) {
    // check for similars shapes of A, B, and dst
    if (A->data->size != B->data->size || A->data->size != dst->data->size) {
        printf("Shape mismatch in tensors!\n");
        return;
    }
    multOp(dst->data, A->data, B->data);
}

void add(Tensor* dst, Tensor* A) {
    // check for similars shapes of A and dst
    if (A->data->size != dst->data->size) {
        printf("Shape mismatch in tensors!\n");
        return;
    }
    addOp(dst->data, A->data);
}

void printTensor(Tensor* A) {
    //printf("%s %d %d", GetDType(A->data->dtype), A->shape[0], A->shape[1]);
    for (int i = 0; i < A->shape[0]; ++i) {
        for (int j = 0; j < A->shape[1]; ++j) {
            int index = i * A->shape[1] + j;
            PrintFunc print_func = print_types[A->data->dtype];
            if (print_func) {
                print_func(A->data->values, index);
            } else {
                printf("Print operation not supported for dtype %d\n", A->data->dtype);
            }
        }
        printf("\n");
    }
}


#endif //TENSOR_IMPLEMENTATION