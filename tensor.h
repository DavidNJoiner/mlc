#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "dtype.h"
#include "ops.h"
#include "debug.h"

#ifndef TENSOR_H_ 
#define TENSOR_H_

typedef struct {
    Data* data;
    float32* gradient;
    int* shape;
    int dim;
    int stride;
} Tensor;


Tensor* tensor(Data* data, bool requires_grad);
Tensor* createTensor(int* shape, int dim, int dtype, bool requires_grad);
Tensor* zerosFrom(Tensor* t);

/*
 * ===========================================================================
 * Prototypes for arithmetic functions on Tensors
 * ===========================================================================
 */
void mult(Tensor* dst, Tensor* A, Tensor* B);
void add(Tensor* dst, Tensor* A);
void freeTensor(Tensor* t);
void print2DTensor(Tensor* A);
void printTensor(Tensor* A);

#endif //TENSOR_H

#ifndef TENSOR_IMPLEMENTATION 
#define TENSOR_IMPLEMENTATION

/*
 * tensor : create a new Tensor from a Data object.
 */
Tensor* tensor(Data* data, bool requires_grad) {
    Tensor* new_tensor = (Tensor*)malloc(sizeof(Tensor));
    if (requires_grad) {
        new_tensor->gradient = (float32*)calloc(data->size, sizeof(float32)); //Currently the gradient array is always of type float32
    }
    new_tensor->data = data;
    new_tensor->shape = new_tensor->data->shape;
    new_tensor->dim = 2;  // Assuming 2 dimensions for simplicity
    new_tensor->stride = new_tensor->data->stride;

    return new_tensor;
}

/*
 * tensor : create a new Tensor from scratch.
 */
Tensor* createTensor(int* shape, int dim, int dtype, bool requires_grad) {
    // Create an array of zeros
    int size = 1;
    for (int i = 0; i < dim; i++) {
        size *= shape[i];
    }
    void* array = calloc(size, dtypeSize(dtype));
    if (array == NULL) {
        printf("Memory allocation failed!\n");
        return NULL;
    }
    
    // Convert array to data
    Data* data = convertToData(array, shape, dim, dtype);
    
    // Create tensor with data
    Tensor* t = tensor(data, requires_grad);
    
    // Free the array
    free(array);

    return t;
}


/*
 * zerosFrom : create a new Tensor filled with zeros from an existing Tensor(template).
 */
Tensor* zerosFrom(Tensor* t) {
    // Allocate new Tensor
    Tensor* new_tensor = (Tensor*)malloc(sizeof(Tensor));

    // Copy properties of the original Tensor
    new_tensor->shape = t->shape;
    new_tensor->dim = t->dim;
    new_tensor->stride = t->stride;

    // Allocate new Data
    Data* new_data = (Data*)malloc(sizeof(Data));

    // Copy properties of the original Data
    new_data->shape = t->data->shape;
    new_data->stride = t->data->stride;
    new_data->dtype = t->data->dtype;
    new_data->size = t->data->size;

    // Allocate zero values
    new_data->values = calloc(new_data->size, sizeof(new_data->dtype));

    // Set the new Data to the new Tensor
    new_tensor->data = new_data;

    // Create gradient for the new tensor if required
    if (t->gradient != NULL) {
        new_tensor->gradient = (float32*)calloc(new_data->size, sizeof(float32));
    } else {
        new_tensor->gradient = NULL;
    }

    return new_tensor;
}

void freeTensor(Tensor* t) {
    if (t != NULL) {
        if (t->gradient != NULL) {
            free(t->gradient);
        }
        if (t->data != NULL) {
            if (t->data->values != NULL) {
                free(t->data->values);
            }
            free(t->data);
        }
        free(t);
    }
}

/*
 * mult : Multiply two Tensors A and B. Stores the result as a third Tensor dst.
 */
void mult(Tensor* dst, Tensor* A, Tensor* B) {
    // check for similars shapes of A, B, and dst
    printf("Hello, Mult!\n");
    if (A->data->shape != B->data->shape || A->data->shape != dst->data->shape) {
        printf("Shape mismatch in tensors!\n");
        return;
    }
    multOp(dst->data, A->data, B->data);
}

/*
 * add : Add two Tensors A and dst. Stores the result in a the Tensor dst.
 */
void add(Tensor* dst, Tensor* A) {
    // check for similars shapes of A and dst
    printf("Hello, Add!\n");
    if (A->data->shape != dst->data->shape) {
        printf("Shape mismatch in tensors!\n");
        return;
    }
    addOp(dst->data, A->data);
}

/*
 * printTensor : print a 2D Tensor to the console.
 */
void print2DTensor(Tensor* A) {
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
    printf("\n");
}

void printTensor(Tensor* A){
    if (0 < A->data->dtype <= 16) {
        printf("let's print ! \n");
        printOp(A->data, A->dim);
    }
}


#endif //TENSOR_IMPLEMENTATION