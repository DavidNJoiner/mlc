#ifndef ARR_H
#define ARR_H

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "../core/types/dtype.h"

// Array Structure
typedef struct Array
{
    void *values;
    int size;
    int dim;
    int *shape;
    int dtype;
} arr_t;

// Array ptr tracker
typedef struct
{
    arr_t **data_ptrs;
    int count;
    int capacity;
} arrPtrTracker_t;

extern arrPtrTracker_t *global_data_ptr_array;
extern int data_total_alloc;
extern int data_total_dealloc;

// Array functions Prototypes
void            arr_flatten(void *array, void *flattened, int *shape, int dim, int dtype, int idx);
void            arr_print(arr_t *dat);

int             compute_index(int *indices, int *strides, int dim);
int             compute_stride(int *shape, int dim, int dtype);
int             compute_size(int *shape, int dim);

arr_t*          arr_create_from_array(void *array, int *shape, int dim, int dtype);
arr_t*          arr_create_from_random(int size, int min_range, int max_range, int *shape, int dim, int dtype);
void*           arr_get_value_at_index(arr_t *data, int *indices);
void            arr_set_value_at_index(arr_t *data, int *indices, void *value);

#endif // ARR_H