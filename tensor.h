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


Tensor*     tensor(Data* data, bool requires_grad);
Tensor*     createTensor(int* shape, int dim, int dtype, bool requires_grad);
Tensor*     zerosFrom(Tensor* t);

/*
   -------------------------------------------------------
   Prototypes for arithmetic functions on Tensors
   -------------------------------------------------------
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
   -------------------------------------------------------
   tensor : create a new Tensor from a Data object.
   -------------------------------------------------------
 */
Tensor* tensor(Data* data, bool requires_grad) {
    Tensor* new_tensor = (Tensor*)malloc(sizeof(Tensor));
    if (requires_grad) {
        new_tensor->gradient = (float32*)calloc(data->size, sizeof(float32)); //Currently the gradient array is always of type float32
    }else{new_tensor->gradient = NULL;}
    new_tensor->data = data;
    new_tensor->shape = new_tensor->data->shape;
    new_tensor->dim = data->dim;
    new_tensor->stride = new_tensor->data->dtype;

    return new_tensor;
}
/*
   -------------------------------------------------------
   tensor : create a new Tensor from scratch.
   -------------------------------------------------------
 */
Tensor* createTensor(int* shape, int dim, int dtype, bool requires_grad) {
    // Create an array of zeros
    int size = 1;
    for (int i = 0; i < dim; i++) {
        size *= shape[i];
    }
    void* array = calloc(size, GetDtypeSize(dtype));
    if (array == NULL) {
        printf("Memory allocation failed!\n");
        return NULL;
    }
    Data* data = convertToData(array, shape, dim, dtype);
    Tensor* t = tensor(data, requires_grad);
    return t;
}
/*
   -------------------------------------------------------
   zerosFrom : create a new Tensor filled with zeros from an existing Tensor(template).
   -------------------------------------------------------
 */
Tensor* zerosFrom(Tensor* t) {
    Tensor* new_tensor = (Tensor*)malloc(sizeof(Tensor));

    new_tensor->shape = (int*)malloc(t->dim * sizeof(int));
    for (int i = 0; i < t->dim; i++) {
        new_tensor->shape[i] = t->shape[i];
    }
    new_tensor->dim = t->dim;
    new_tensor->stride = t->stride;

    Data* new_data = (Data*)malloc(sizeof(Data));

    new_data->shape = (int*)malloc(t->data->dim * sizeof(int));
    for (int i = 0; i < t->data->dim; i++) {
        new_data->shape[i] = t->data->shape[i];
    }
    new_data->dim = t->data->dim;
    new_data->size = t->data->size;
    new_data->dtype = t->data->dtype;

    switch (new_data->dtype) {
        case FLOAT32:
            new_data->values = (float32 *)aligned_alloc(32, sizeof(float));
            break;
        case FLOAT64:
            new_data->values = (float64 *)aligned_alloc(32, sizeof(double));
            break;
        default:
            fprintf(stderr, "Unsupported dtype: %d\n", new_data->dtype);
            exit(EXIT_FAILURE);
    }

    new_tensor->data = new_data;

    if (t->gradient != NULL) {
        new_tensor->gradient = (float32*)calloc(new_data->size, sizeof(float32));
    } else {
        new_tensor->gradient = NULL;
    }

    return new_tensor;
}
/*
   -------------------------------------------------------
   freeTensor : Releases the memory allocated for a given tensor,.
   -------------------------------------------------------
 */
void freeTensor(Tensor* t) {
    if (t != NULL) {
        if (t->gradient != NULL) {
            free(t->gradient);
            t->gradient = NULL;
        }
        if (t->data != NULL) {
            if (t->data->values != NULL) {
                free(t->data->values);
                t->data->values = NULL; 
            }
            free(t->data);
            t->data = NULL;  
        }
        free(t);
        t = NULL;
    }
}
/*
   -------------------------------------------------------
   shapesAreEqual : Check if two Tensors shapes are equals.
   -------------------------------------------------------
 */
bool shapesAreEqual(Tensor* A, Tensor* B) {
    if (A->dim != B->dim) {
        return false;
    }

    for (int i = 0; i < A->dim; i++) {
        if (A->shape[i] != B->shape[i]) {
            return false;
        }
    }

    return true;
}
/*
   -------------------------------------------------------
   mult : Multiply two Tensors A and B. Stores the result as a third Tensor dst.
   -------------------------------------------------------
 */
void mult(Tensor* dst, Tensor* A, Tensor* B) {

    if (!shapesAreEqual(A, B) || !shapesAreEqual(A, dst)) {
        printf("Shape mismatch in tensors!\n");
        return;
    }

    multOp(dst->data, A->data, B->data);
}
/*
   -------------------------------------------------------
   fastmul : Multiply two Tensors A and B. Stores the result as a third Tensor dst (only float32).
   -------------------------------------------------------
 */
void fastmult(Tensor* dst, Tensor* A, Tensor* B) {

    if (!shapesAreEqual(A, B) || !shapesAreEqual(A, dst)) {
        printf("Shape mismatch in tensors!\n");
        return;
    }

    if (is_aligned(dst->data->values, 32) && is_aligned(A->data->values, 32) && is_aligned(B->data->values, 32)) {
        gemmOp(dst->data, A->data, B->data);
    } else {
        printf("values are NOT 32-byte aligned.\n");
    }
}
/*
   -------------------------------------------------------
   add : Add two Tensors A and dst. Stores the result in a the Tensor dst.
   -------------------------------------------------------
 */
void add(Tensor* dst, Tensor* A) {

    if (!shapesAreEqual(A, dst)) {
        printf("Shape mismatch in tensors!\n");
        return;
    }

    addOp(dst->data, A->data);
}
/*
   -------------------------------------------------------
   printTensor : print a Tensor to the console.
   -------------------------------------------------------
 */
void printTensor(Tensor* A){
    if (0 < A->data->dtype <= 16) {
        printf("\ndebug values_ptr : %p \n", A->data->values);
        printf("debug dtype : %d \n", A->data->dtype);
        printf("debug shape : ");
        for (int i = 0; i < A->dim; i++) {
            printf("%d ", A->shape[i]);
        }
        printf("\n");
        printf("debug dim : %d \n \n", A->dim);
        printOp(A->data, A->dim);
    }
}


#endif //TENSOR_IMPLEMENTATION