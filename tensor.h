#ifndef TENSOR_H_ 
#define TENSOR_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#include "dtype.h"
#include "debug.h"
#include "ops.h"

//#include "device_manager.h"

typedef struct {
    Data* data;
    float32* gradient;
} Tensor;

Tensor*     tensor(Data* data, bool requires_grad);
Tensor*     createTensor(int* shape, int dim, int dtype, bool requires_grad);
Tensor*     zerosFrom(Tensor* t);
Tensor*     new_full(int* shape, int fill_value, int dtype, bool requires_grad);

//  Tensors arithmetic
void mult(Tensor* dst, Tensor* A, Tensor* B);
void avx_mul(Tensor* dst, Tensor* A, Tensor* B);
void add(Tensor* dst, Tensor* A);
void avx_add(Tensor* dst, Tensor* A);

//  Tensors modifications.
void freeTensor(Tensor* t);
//transpose here

void printTensor(Tensor* A);

#endif //TENSOR_H

#ifndef TENSOR_IMPLEMENTATION 
#define TENSOR_IMPLEMENTATION

/*  -------------------------------------------------------*/
/*  tensor : create a new Tensor from a Data object.       */
/*  -------------------------------------------------------*/
Tensor* tensor(Data* data, bool requires_grad) {
    Tensor* new_tensor = (Tensor*)malloc(sizeof(Tensor));
    if (requires_grad) {
        new_tensor->gradient = (float32*)calloc(data->size, sizeof(float32)); //Currently the gradient array is always of type float32
    }else{new_tensor->gradient = NULL;}
    new_tensor->data = data;
    return new_tensor;
}
/*  -------------------------------------------------------*/
/*  createTensor : create a new Tensor from scratch.       */
/*  -------------------------------------------------------*/
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
/*  -------------------------------------------------------------------------------------*/
/*  zerosFrom : create a new Tensor filled with zeros from an existing Tensor(template). */
/*  -------------------------------------------------------------------------------------*/
Tensor* zerosFrom(Tensor* t) {
    Tensor* new_tensor = (Tensor*)malloc(sizeof(Tensor));
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
            new_data->values = (float32 *)aligned_alloc(32, new_data->size * sizeof(float32));
            break;
        case FLOAT64:
            new_data->values = (float64 *)aligned_alloc(32, new_data->size * sizeof(float64));
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

Tensor* new_full(int* shape, int fill_value, int dtype, bool requires_grad){
    
} 
/*  ---------------------------------------------------------------*/
/*  freeTensor : Releases the memory allocated for a given tensor. */
/*  ---------------------------------------------------------------*/
void freeTensor(Tensor* t) {
    if (t != NULL) {
    if (t->data != NULL) {
        if (t->data->values != NULL) {
            free(t->data->values);
            t->data->values = NULL;
        }
        free(t->data);
        t->data = NULL;
    }
    if (t->gradient != NULL) {
        free(t->gradient);
        t->gradient = NULL;
    }
    free(t);
    t = NULL;
    }
}
/*  ---------------------------------------------------------------*/
/*  shapesAreEqual : Check if two Tensors shapes are equals.       */
/*  ---------------------------------------------------------------*/
bool shapesAreEqual(Tensor* A, Tensor* B) {
    if (A->data->dim != B->data->dim) {
        printf("Dim mismatch in tensors!\n");
        return false;
    }

    for (int i = 0; i < A->data->dim; i++) {
        if (A->data->shape[i] != B->data->shape[i]) {
            printf("Shape mismatch in tensors! ");
            return false;
        }
    }

    return true;
}
/*  -------------------------------------------------------------------------------*/
/*  mult : Multiply two Tensors A and B. Stores the result as a third Tensor dst.  */
/*  -------------------------------------------------------------------------------*/
void mult(Tensor* dst, Tensor* A, Tensor* B) {

    if (!shapesAreEqual(A, B) || !shapesAreEqual(A, dst)) {
        return;
    }

    multOp(dst->data, A->data, B->data);
}
/*  --------------------------------------------------------------------------------*/
/*  fastmul : Multiply two Tensors A and B. Stores the result as a third Tensor dst */
/*  --------------------------------------------------------------------------------*/
void avx_mul(Tensor* dst, Tensor* A, Tensor* B) {

    if (!shapesAreEqual(A, B) || !shapesAreEqual(A, dst)) {
        return;
    }

    if (is_aligned(dst->data->values, 32) && is_aligned(A->data->values, 32) && is_aligned(B->data->values, 32)) {
        avx_mul_op(dst->data, A->data, B->data);
    } else {
        printf("values are NOT 32-byte aligned. Running scalar mult\n");
        mult(dst, A, B);
    }
}
/*  -------------------------------------------------------------------------*/
/*   add : Add two Tensors A and dst. Stores the result in a the Tensor dst. */
/*  -------------------------------------------------------------------------*/
void add(Tensor* dst, Tensor* A) {

    if (!shapesAreEqual(A, dst)) {
        return;
    }

    addOp(dst->data, A->data);
}
/*  -----------------------------------------------------------------------------*/
/*  avx_add : Add two Tensors A and B. Stores the result as a third Tensor dst   */
/*  -----------------------------------------------------------------------------*/
void avx_add(Tensor* dst, Tensor* A) {

    if (!shapesAreEqual(A, dst)) {
        return;
    }

    if (is_aligned(dst->data->values, 32) && is_aligned(A->data->values, 32)) {
        avx_add_op(dst->data, A->data);
    } else {
        printf("values are NOT 32-byte aligned. Running scalar mult\n");
        add(dst, A);
    }
}
/*  ---------------------------------------------------------------*/
/*   printTensor : print a Tensor to the console.                  */
/*  ---------------------------------------------------------------*/
void printTensor(Tensor* A) {

    if(A == NULL) {
        printf("Error: Null Tensor pointer passed to printTensor function.\n");
        return;
    }

    if(A->data == NULL) {
        printf("Error: Null Data pointer inside the Tensor structure.\n");
        return;
    }

    if(A->data->shape == NULL) {
        printf("Error: Null Shape pointer inside the Tensor structure.\n");
        return;
    }

    if (0 < A->data->dtype && A->data->dtype <= 16) {
        printData(A->data);
    }else {
        printf("Error: Invalid dtype value.\n");
    }
}


#endif //TENSOR_IMPLEMENTATION