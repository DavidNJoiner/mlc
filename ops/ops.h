#ifndef OPS_H_
#define OPS_H_

#include "../data/arr.h"
#include "../core/config.h"
#include "../core/device.h"

void intel_mul_1D(arr_t *dst, arr_t *A, arr_t *B, Device *device);
void intel_add_1D(arr_t *dst, arr_t *A, Device *device);

#endif // OPS_H