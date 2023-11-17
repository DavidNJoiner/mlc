#include "arr.h"
#include "../debug.h"
#include "../core/mempool/mempool.h"

arrPtrTracker_t * global_data_ptr_array = NULL;
int data_total_alloc = 0;
int data_total_dealloc = 0;

/* Converts multi-dimensional index into a linear index. */
int compute_index(int *indices, int *shape, int dim)
{
    int index = 0;
    int stride = 1;

    for (int i = dim - 1; i >= 0; i--)
    {
        index += indices[i] * stride;
        stride *= shape[i];
    }
    return index;
}

int compute_stride(int *shape, int dim, int dtype)
{
    int stride = 1;
    for (int i = 1; i < dim; i++)
    {
        stride *= shape[i];
    }
    stride *= get_data_size(dtype); // compute the stride for the current dimension
    return stride;
}

int compute_size(int *shape, int dim)
{
    int size = 1;
    for (uint32_t i = 0; i < dim; i++)
    {
        size *= shape[i];
    }
    return size;
}

/* Recursively flattens a multi-dimensional array into a one-dimensional array. */
void arr_flatten(void *array, void *flattened, int *shape, int dim, int dtype, int idx)
{
    if (dim == 0)
    {
        switch (dtype)
        {
        case FLOAT16:
        {
            float16 *farray = (float16 *)array;
            float16 *fflattened = (float16 *)flattened;
            fflattened[idx] = *farray;
            break;
        }
        case FLOAT32:
        {
            float32 *farray = (float32 *)array;
            float32 *fflattened = (float32 *)flattened;
            fflattened[idx] = *farray;
            break;
        }
        case FLOAT64:
        {
            float64 *farray = (float64 *)array;
            float64 *fflattened = (float64 *)flattened;
            fflattened[idx] = *farray;
            break;
        }
        default:
            printf("Failed to flatten : Unsupported dtype %d\n", dtype);
            break;
        }
    }
    else
    {
        // recursive case: move to the next dimension
        int stride = compute_stride(shape, dim, dtype);
        for (uint32_t i = 0; i < shape[0]; i++)
        {
            arr_flatten((char *)array + i * stride, flattened, shape + 1, dim - 1, dtype, idx + i * stride / get_data_size(dtype));
        }
    }
}

/* Converts a given multi-dimensional array into a Array structure. */
arr_t *arr_create_from_array(void *source_array, int *shape, int dim, int dtype)
{
    int size = compute_size(shape, dim);
    void* mem_ptr = memory_alloc_padded (size, dtype);
    arr_flatten(source_array, mem_ptr, shape, dim, dtype, 0);

    arr_t *dat = arr_alloc();

    dat->values = mem_ptr;
    dat->size = size;
    dat->dim = dim;
    dat->shape = shape;
    dat->dtype = dtype;

    data_total_alloc += sizeof(arr_t);
    arr_increment_ptr_count(dat);

    return dat;
}

/* Display properties of the Array structure. */
void arr_print(arr_t *dat) {
    const char *dtypeStr = get_data_type(dat->dtype);
    if (dtypeStr == NULL) {
        printf("Error: getDType returned NULL.\n");
        return;
    }

    printf("dtype : %4s \n", dtypeStr);
    printf("shape : ");
    for (uint32_t i = 0; i < dat->dim; i++) {
        printf("[%2d] ", dat->shape[i]);
    }
    printf("\n");
    printf("dimension : %d \n \n", dat->dim);

    if (dat->dim < 0) {
        printf("Error: Invalid dimension for printOp.\n");
        return;
    }

    PrintOp(dat, dat->dim);
}

/* Create a Array object filled with random values within a given range. */
arr_t *arr_create_from_random(int size, int min_range, int max_range, int *shape, int dim, int dtype)
{
    srand((unsigned int)time(NULL)); // Seed the random number generator

    void* mem_ptr = memory_alloc_padded (size, dtype);
    arr_t* data = arr_alloc();

    switch (dtype) {
        case FLOAT16:
            for (int i = 0; i < size; i++) {
                float random_value = min_range + ((float)rand() / (float)RAND_MAX) * (max_range - min_range);
                ((float16 *)mem_ptr)[i] = float16_from_float(random_value);
            }
            break;
        case FLOAT32:
            for (int i = 0; i < size; i++) {
                ((float32 *)mem_ptr)[i] = (float32)(min_range + ((float32)rand() / (float32)RAND_MAX) * (max_range - min_range));
            }
            break;
        case FLOAT64:
            for (int i = 0; i < size; i++) {
                ((float64 *)mem_ptr)[i] = (float64)(min_range + ((float64)rand() / (float64)RAND_MAX) * (max_range - min_range));
            }
            break;
        default:
            printf("Unsupported dtype for random fill %s\n", get_data_type(dtype));
            break;
    }

    data->values = mem_ptr;
    data->size = size;
    data->dim = dim;
    data->shape = shape;
    data->dtype = dtype;

    data_total_alloc += sizeof(arr_t);
    arr_increment_ptr_count(data);

    return data;
}

/* Access an element in the Array object. */
void *arr_get_value_at_index(arr_t *data, int *indices) {
    int linear_index = compute_index(indices, data->shape, data->dim);
    int dtype_size = get_data_size(data->dtype);

    // Cast the data values to the correct type and return the address of the element
    switch (data->dtype) {
        case FLOAT16:
            return (void *)((float16 *)data->values + linear_index);
        case FLOAT32:
            return (void *)((float32 *)data->values + linear_index);
        case FLOAT64:
            return (void *)((float64 *)data->values + linear_index);
        default:
            printf("Unsupported dtype %s\n", get_data_type(data->dtype));
            return NULL;
    }
}

/* Set the value of an element in the Array object. */
void arr_set_value_at_index(arr_t *data, int *indices, void *value) {
    void *element_ptr = arr_get_value_at_index(data, indices);

    switch (data->dtype) {
        case FLOAT16:
            *((float16 *)element_ptr) = *((float16 *)value);
            break;
        case FLOAT32:
            *((float32 *)element_ptr) = *((float32 *)value);
            break;
        case FLOAT64:
            *((float64 *)element_ptr) = *((float64 *)value);
            break;
        default:
            printf("Failed to set element: Unsupported dtype %s\n", get_data_type(data->dtype));
            break;
    }
}

