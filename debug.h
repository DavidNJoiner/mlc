#include "dtype.h"

#ifndef DEBUG_H_ 
#define DEBUG_H_

typedef void (*PrintFunc)(void*, int);

void print_float32(void* values, int index);
void print_float64(void* values, int index);

void printHelper(Data* A, PrintFunc printFunc, int* indices, int dim, int cur_dim);
void printOp(Data* A, int dim);

#endif //DEBUG_H

#ifndef DEBUG_IMPLEMENTATION
#define DEBUG_IMPLEMENTATION

/*  
    -------------------------------------------------------
    Print Ops
    -------------------------------------------------------
*/ 
void print_float16(void* values, int index) {
    float16* vals = (float16*)values;
    printf("%.2hu \t", vals[index]);
}
void print_float32(void* values, int index) {
    float32* vals = (float32*)values;
    printf("%.2f \t", vals[index]);
}
void print_float64(void* values, int index) {
    float64* vals = (float64*)values;
    printf("%.4lf \t", vals[index]);
}

/*  
    -------------------------------------------------------
    PRINT Ops lookup
    -------------------------------------------------------
*/
PrintFunc print_types[] = {
    [FLOAT32] = print_float32,
    [FLOAT64] = print_float64,
    [FLOAT16] = print_float16,
};

/*  
    -------------------------------------------------------
    printHelper : Recursive helper function to print array pointed to by a Data struct
    -------------------------------------------------------
*/
void printHelper(Data* A, PrintFunc printFunc, int* indices, int dim, int cur_dim) {
    if (cur_dim == dim - 1) {
        for (indices[cur_dim] = 0; indices[cur_dim] < A->shape[cur_dim]; indices[cur_dim]++) {
            int index = calculateIndex(indices, A->shape, dim);
            printFunc(A->values, index);
        }
        printf("\n");
    } else {
        for (indices[cur_dim] = 0; indices[cur_dim] < A->shape[cur_dim]; indices[cur_dim]++) {
            printHelper(A, printFunc, indices, dim, cur_dim + 1);
        }
    }
}

/*  
    -------------------------------------------------------
    printOp : Print any dtype Tensor of any dimension to the console.
    -------------------------------------------------------
*/
void printOp(Data* A, int dim) {
    PrintFunc printFunc = print_types[A->dtype];
    if (printFunc) {
        int* indices = (int*)calloc(dim, sizeof(int));
        printHelper(A, printFunc, indices, dim, 0);
        free(indices);
    } else {
        printf("Cannot print dtype %d\n", A->dtype);
    }
}

#endif //DEBUG_IMPLEMENTATION
