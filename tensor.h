#ifndef TENSOR_H_ 
#define TENSOR_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdbool.h>
#include <math.h>

//#include "memory_pool.h"
#include "data.h"
#include "debug.h"
#include "ops.h"

#include "device_manager.h"
#include "device.h"

/*  -------------------------------------------------------*/
/*  Tensor memory management                               */
/*  -------------------------------------------------------*/
// Note: Memory management in autograd systems is complex.
// For example, when freeing a Tensor, we should consider whether the gradient field
// needs to be freed as well, as it might still be needed by other Tensors.
// Additionally, the creator field should be freed if dynamically allocating Function objects.

// When implementing a deep learning framework, it's important to explore
// more efficient memory handling techniques than individual Tensor allocation
// and freeing. One approach is to allocate large memory blocks and partition
// them for individual Tensors, or to reuse memory from Tensors that are no longer needed.
// These strategies can significantly improve performance by reducing memory overhead,
// but they also introduce greater complexity in memory management.

typedef struct {
    bool require_grad;
    Data* data;
    Device* device;
    float32* gradient;
    //LazyBuffer* lazydata;
    //Function* creator; // Points to the Function that created this Tensor.
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
void freeTensor(Tensor** t);
//transpose here

void printTensor(Tensor* A);
bool is_aligned(void* ptr, size_t alignment);

#endif //TENSOR_H