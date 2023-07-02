#ifndef TENSOR_H_ 
#define TENSOR_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdbool.h>
#include <math.h>
#include "data.h"
#include "debug.h"
#include "ops.h"

#include "device_manager.h"
#include "device.h"

typedef struct {
    Data* data;
    Device* device;
    float32* gradient;
} Tensor;

//  Tensors creation
Tensor*     zerosFrom(Tensor* t);
Tensor*     tensor(Data* data, Device* device, bool requires_grad);
Tensor*     createTensor(int* shape, int dim, int dtype, Device* device, bool requires_grad);
Tensor*     newFull(int* shape, int fill_value, int dtype, Device* device, bool requires_grad);

//  Tensors arithmetic
void mul(Tensor* dst, Tensor* A, Tensor* B);
void add(Tensor* dst, Tensor* A);


//  Tensors modifications.
void freeTensor(Tensor* t);
//transpose here

void printTensor(Tensor* A);

#endif //TENSOR_H