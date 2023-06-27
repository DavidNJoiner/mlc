#ifndef OPS_H_ 
#define OPS_H_

#include "dtype.h"
#include "avx.h"

// Arithmetic
typedef void (*MultFunc)(void*, void*, void*, int);
typedef void (*AddFunc)(void*, void*, int);

#define DEFINE_OPS(TYPE) \
void add_##TYPE(void* dstValues, void* AValues, int size) { \
    TYPE* AFloat = (TYPE*)AValues; \
    TYPE* dstFloat = (TYPE*)dstValues; \
    for (int i = 0; i < size; i++) { \
        dstFloat[i] += AFloat[i]; \
    } \
} \
\
void mult_##TYPE(void* dstValues, void* AValues, void* BValues, int size) { \
    TYPE* AFloat = (TYPE*)AValues; \
    TYPE* BFloat = (TYPE*)BValues; \
    TYPE* dstFloat = (TYPE*)dstValues; \
    for (int i = 0; i < size; i++) { \
        dstFloat[i] = AFloat[i] * BFloat[i]; \
    } \
}

DEFINE_OPS(float32)
DEFINE_OPS(float64)
DEFINE_OPS(float16)

void add_op(Data* dst, Data* A);
void mul_op(Data* dst, Data* A, Data* B);
void avx_add_op(Data* dst, Data* A);
void avx_mul_op(Data* dst, Data* A, Data* B);

#endif //OPS_H

#ifndef OPS_IMPLEMENTATION
#define OPS_IMPLEMENTATION

AddFunc addData[] = {
    [FLOAT32] = add_float32,
    [FLOAT64] = add_float64,
    [FLOAT16] = add_float16,
};

MultFunc multData[] = {
    [FLOAT32] = mult_float32,
    [FLOAT64] = mult_float64,
    [FLOAT16] = mult_float16,
};
/*  -------------------------------------------------------*/
/*  add_op : Tensor Add Operation.                         */
/*  -------------------------------------------------------*/
void add_op(Data* dst, Data* A) {
    if (A->dtype != dst->dtype) {
        printf("Data dtypes do not match!\n");
        return;
    }

    if (A->dtype < 0 || A->dtype >= sizeof(addData) / sizeof(addData[0])) {
        printf("Invalid dtype!\n");
        return;
    }

    AddFunc addFunc = addData[A->dtype];
    if (addFunc) {
        addFunc(dst->values, A->values, A->size);
    } else {
        printf("Operation not supported for dtype %d\n", A->dtype);
    }
}
/*  -------------------------------------------------------*/
/*  mul_op : Tensor Multiply Operation.                    */
/*  -------------------------------------------------------*/
void mul_op(Data* dst, Data* A, Data* B) {
    if (A->dtype != dst->dtype || B->dtype != dst->dtype ) {
        printf("Data dtypes do not match!\n");
        return;
    }

    if (A->dtype < 0 || A->dtype >= sizeof(multData) / sizeof(multData[0])) {
        printf("Invalid dtype!\n");
        return;
    }

    MultFunc multFunc = multData[A->dtype];
    if (multFunc) {
        multFunc(dst->values, A->values, B->values, A->size);
    } else {
        printf("Operation not supported for dtype %d\n", A->dtype);
    }
}
/*  -------------------------------------------------------*/
/*  avx_mul_op : Tensor Fast Multiply Operation.           */
/*  -------------------------------------------------------*/
void avx_mul_op(Data* dst, Data* A, Data* B){
    int mat_size = dst->size;
    switch (dst->dtype) {
        case FLOAT32: 
            vec1_avx_mul_float32(dst->values, A->values, B->values, mat_size);
            break;
        case FLOAT64: 
            vec1_avx_mul_float64(dst->values, A->values, B->values, mat_size);
            break;
    }
}
/*  -------------------------------------------------------*/
/*  avx_add_op : Tensor Fast Add Operation.                */
/*  -------------------------------------------------------*/
void avx_add_op(Data* dst, Data* A){
    int mat_size = dst->size;
    switch (dst->dtype) {
        case FLOAT32: 
            vec1_avx_add_float32(dst->values, A->values, mat_size);
            break;
        case FLOAT64: 
            vec1_avx_add_float64(dst->values, A->values, mat_size);
            break;
    }
}

#endif //OPS_IMPLEMENTATION
