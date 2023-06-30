 #ifndef DEBUG_H_ 
#define DEBUG_H_

#include <stdbool.h> 
#include <stdint.h> 
#include <stddef.h> 
#include "data.h"

typedef void (*PrintFunc)(void*, int);

bool is_aligned(void* ptr, size_t alignment);

uint64_t nanos();

void print_float16(void* values, int index);
void print_float32(void* values, int index);
void print_float64(void* values, int index);

void printHelper(Data* A, PrintFunc printFunc, int* indices, int dim, int cur_dim);
void print_op(Data* A, int dim);

#endif //DEBUG_H
