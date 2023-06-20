#include "dtype.h"

#ifndef OPS_H_ 
#define OPS_H_

//Arithmetic
typedef void (*MultFunc)(void*, void*, void*, int);
typedef void (*AddFunc)(void*, void*, int);

void add_float32(void* dstValues, void* AValues, int size);
void mult_float32(void* dstValues, void* AValues, void* BValues, int size);
void add_float64(void* dstValues, void* AValues, int size);
void mult_float64(void* dstValues, void* AValues, void* BValues, int size);

void addOp(Data* dst, Data* A);
void multOp(Data* dst, Data* A, Data* B);

#endif //OPS_H

#ifndef OPS_IMPLEMENTATION
#define OPS_IMPLEMENTATION

/*  
    -------------------------------------------------------
    float32 Ops
    -------------------------------------------------------
*/ 
void add_float32(void* dstValues, void* AValues, int size) {
    float32* AFloat = (float32*)AValues;
    float32* dstFloat = (float32*)dstValues;
    for (int i = 0; i < size; i++) {
        dstFloat[i] += AFloat[i];
    }
}

void mult_float32(void* dstValues, void* AValues, void* BValues, int size) {
    float32* AFloat = (float32*)AValues;
    float32* BFloat = (float32*)BValues;
    float32* dstFloat = (float32*)dstValues;
    for (int i = 0; i < size; i++) {
        dstFloat[i] = AFloat[i] * BFloat[i];
    }
}

/*  
    -------------------------------------------------------
    float64 Ops
    -------------------------------------------------------
*/ 
void add_float64(void* dstValues, void* AValues, int size) {
    float64* AFloat = (float64*)AValues;
    float64* dstFloat = (float64*)dstValues;
    for (int i = 0; i < size; i++) {
        dstFloat[i] += AFloat[i];
    }
}

void mult_float64(void* dstValues, void* AValues, void* BValues, int size) {
    float64* AFloat = (float64*)AValues;
    float64* BFloat = (float64*)BValues;
    float64* dstFloat = (float64*)dstValues;
    for (int i = 0; i < size; i++) {
        dstFloat[i] = AFloat[i] * BFloat[i];
    }
}

/*  
    -------------------------------------------------------
    float16 Ops
    -------------------------------------------------------
*/ 
void add_float16(void* dstValues, void* AValues, int size) {
    float16* AFloat = (float16*)AValues;
    float16* dstFloat = (float16*)dstValues;
    for (int i = 0; i < size; i++) {
        dstFloat[i] += AFloat[i];
    }
}

void mult_float16(void* dstValues, void* AValues, void* BValues, int size) {
    float16* AFloat = (float16*)AValues;
    float16* BFloat = (float16*)BValues;
    float16* dstFloat = (float16*)dstValues;
    for (int i = 0; i < size; i++) {
        dstFloat[i] = AFloat[i] * BFloat[i];
    }
}

/*  
    -------------------------------------------------------
    ADD Ops lookup
    -------------------------------------------------------
*/ 
AddFunc addData[] = {
    [FLOAT32] = add_float32,
    [FLOAT64] = add_float64,
    [FLOAT16] = add_float16,
};

/*  
    -------------------------------------------------------
    MULT Ops lookup
    -------------------------------------------------------
*/ 
MultFunc multData[] = {
    [FLOAT32] = mult_float32,
    [FLOAT64] = mult_float64,
    [FLOAT16] = mult_float16,
};

// Add a dst and A Data objects element-wise and store the result in the res Data object.
void addOp(Data* dst, Data* A) {
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

// Multiply A and B Data objects element-wise and store the result in a dst Data object..
void multOp(Data* dst, Data* A, Data* B) {
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
#endif //OPS_IMPLEMENTATION
