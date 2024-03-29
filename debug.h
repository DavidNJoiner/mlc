#ifndef DEBUG_H_
#define DEBUG_H_

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include "./data/arr.h"

typedef void (*PrintFunc)(void *, int);

bool is_tensor_aligned(void *ptr, size_t alignment);

void print_float16(void *values, int index);
void print_float32(void *values, int index);
void print_float64(void *values, int index);

void PrintArray(void *array, PrintFunc printFunc, int *shape, int dim, int dtype, int idx);
void PrintOp(arr_t *A, int dim);

#endif // DEBUG_H
