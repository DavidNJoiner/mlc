#ifndef DATA_H
#define DATA_H

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "../core/types/dtype.h"

// Data Structure
typedef struct
{
    void *values;
    int size;
    int dim;
    int *shape;
    int dtype;
} Data;

// Data ptr tracker
typedef struct
{
    Data **data_ptrs;
    int count;
    int capacity;
} DataPtrArray;

extern DataPtrArray *global_data_ptr_array;
extern int data_total_alloc;
extern int data_total_dealloc;

// Data functions Prototypes
void array_flatten(void *array, void *flattened, int *shape, int dim, int dtype, int idx);
void data_print(Data *dat);

int compute_index(int *indices, int *strides, int dim);
int compute_stride(int *shape, int dim, int dtype);
int compute_size(int *shape, int dim);

Data *data_create_from_array(void *array, int *shape, int dim, int dtype);
Data *data_create_from_random(int size, int min_range, int max_range, int *shape, int dim, int dtype);
void *data_get_element_at_index(Data *data, int *indices);
void data_set_element_at_index(Data *data, int *indices, void *value);
void fill_data_create_from_random(void *array, int dtype, int size, int min_range, int max_range);

#endif // DATA_H