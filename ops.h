#ifndef OPS_H_ 
#define OPS_H_

#include "data/dataset.h"
#include "config.h"
#include "device.h"
#include "cuda_ops.h"
#include "avx_ops.h"

void speed_mul_op(Data* dst, Data* A, Data* B, Device* device);
void speed_add_op(Data* dst, Data* A, Device* device);

#endif //OPS_H