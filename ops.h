#ifndef OPS_H_ 
#define OPS_H_

#include "config.h"
#include "data.h"
#include "cuda.h"
#include "avx.h"

void speed_mul_op(Data* dst, Data* A, Data* B);
void speed_add_op(Data* dst, Data* A);

#endif //OPS_H