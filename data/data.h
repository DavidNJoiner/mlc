#ifndef DATA_H
#define DATA_H

#include <stdlib.h>
#include <stdio.h>
#include <time.h> 
#include "../dtype.h"

// Data Structure
typedef struct {
    void* values;
    int size;
    int dim;
    int* shape;
    int dtype;
} Data;

// Data ptr tracker
typedef struct {
    Data** data_ptrs;
    int count;
    int capacity;
} DataPtrArray;

extern DataPtrArray* global_data_ptr_array;
extern int total_data_allocated;
extern int total_data_deallocated;

// Data functions Prototypes
void            flatten_array(void* array, void* flattened, int* shape, int dim, int dtype, int idx);
void            display_data(Data* dat);
int             calculate_index(int* indices, int* strides, int dim);
int             calculate_stride(int* shape, int dim, int dtype);
int             calculate_size(int* shape, int dim);
Data*           create_data(void* array, int* shape, int dim, int dtype);
Data*           random_data(int size, int min_range, int max_range, int* shape, int dim, int dtype);
void*           get_data_element(Data* data, int* indices);
void            set_data_element(Data* data, int* indices, void* value);
void            fill_random_data(void* array, int dtype, int size, int min_range, int max_range);


#endif // DATA_H