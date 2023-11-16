#ifndef OPS_H_
#define OPS_H_

#include "../data/arr.h"
#include "../core/config.h"
#include "../core/device.h"

//ops includes
#include "cuda_ops.h"
#include "intrinsics.h"

void speed_mul_op(arr_t *dst, arr_t *A, arr_t *B, Device *device);
void speed_add_op(arr_t *dst, arr_t *A, Device *device);

#endif // OPS_H