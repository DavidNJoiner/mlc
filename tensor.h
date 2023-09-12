#ifndef TENSOR_H_
#define TENSOR_H_

// std
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdbool.h>
#include <math.h>

#include "data/dataset.h"
#include "debug.h"
#include "ops/ops.h"

#include "device_manager.h"
#include "core/device.h"

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

struct Pool_t;

typedef struct
{
    float32 *gradient; // 1 flag, 31 bits gradient
    Data *data;
    Device *device;
    void *lazy;    // LazyBuffer* lazydata;
    void *creator; // Function* creator; Points to the Function that created this Tensor.
} Tensor;

// Utils
void set_require_grad(Tensor *tensor, int bit_flag);
bool get_require_grad(Tensor *tensor);

//  Tensors creation
Tensor *zerosFrom(Tensor *t);
Tensor *tensor(Data *data, Device *device, bool requires_grad);
Tensor *create_tensor(int *shape, int dim, int dtype, Device *device, bool requires_grad);
Tensor *newFull(int *shape, int fill_value, int dtype, Device *device, bool requires_grad);

//  Tensors arithmetic
void mul(Tensor *dst, Tensor *A, Tensor *B);
void add(Tensor *dst, Tensor *A);

// transpose here

void displayTensor(Tensor *A);
bool is_aligned(void *ptr, size_t alignment);

#endif // TENSOR_H