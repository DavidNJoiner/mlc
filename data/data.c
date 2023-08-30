#include "data.h"
#include "../debug.h"
#include "../core/mempool/mempool.h"

DataPtrArray *global_data_ptr_array = NULL;
int total_data_allocated = 0;
int total_data_deallocated = 0;

/* Converts multi-dimensional index into a linear index. */
int calculate_index(int *indices, int *shape, int dim)
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

int calculate_stride(int *shape, int dim, int dtype)
{
    int stride = 1;
    for (int i = 1; i < dim; i++)
    {
        stride *= shape[i];
    }
    stride *= get_data_size(dtype); // calculate the stride for the current dimension
    return stride;
}

int calculate_size(int *shape, int dim)
{
    int size = 1;
    for (uint32_t i = 0; i < dim; i++)
    {
        size *= shape[i];
    }
    return size;
}

/* Recursively flattens a multi-dimensional array into a one-dimensional array. */
void flatten_array(void *array, void *flattened, int *shape, int dim, int dtype, int idx)
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
        int stride = calculate_stride(shape, dim, dtype);
        for (uint32_t i = 0; i < shape[0]; i++)
        {
            flatten_array((char *)array + i * stride, flattened, shape + 1, dim - 1, dtype, idx + i * stride / get_data_size(dtype));
        }
    }
}

/* Converts a given multi-dimensional array into a Data structure. */
Data *create_data(void *array, int *shape, int dim, int dtype)
{
    int size = calculate_size(shape, dim);
    int byte_size = size * get_data_size(dtype);
    void *flattened = (float32 *)aligned_alloc(32, byte_size);

    flatten_array(array, flattened, shape, dim, dtype, 0);

    Data *dat = (Data *)malloc(sizeof(Data));
    dat->values = flattened;
    dat->size = size;
    dat->dim = dim;
    dat->shape = shape;
    dat->dtype = dtype;

    total_data_allocated += sizeof(Data);
    add_data_ptr(dat); // Store pointer to that Data object

    return dat;
}

/* Display properties of the Data structure. */
void display_data(Data *dat)
{
    const char *dtypeStr = get_data_type(dat->dtype);
    if (dtypeStr == NULL)
    {
        printf("Error: getDType returned NULL.\n");
        return;
    }

    printf("dtype : %4s \n", dtypeStr);
    printf("shape : ");
    for (uint32_t i = 0; i < dat->dim; i++)
    {
        printf("[%2d] ", dat->shape[i]);
    }
    printf("\n");
    printf("dimension : %d \n \n", dat->dim);

    if (dat->dim >= 0)
    {
        PrintOp(dat, dat->dim);
    }
    else
    {
        printf("Error: Invalid dimension for printOp.\n");
    }
}

/* Generate a Data object filled with random values. */
Data *random_data(int size, int min_range, int max_range, int *shape, int dim, int dtype)
{
    srand((unsigned int)time(NULL)); // Seed the random number generator

    int dtypeSize = get_data_size(dtype);
    int byte_size = size * dtypeSize;
    int alignment = 32; // Was 32 by default
    const char *dtypeName = get_data_type(dtype);

    // Making sure byte_size is a multiple of the alignment
    if (byte_size % alignment != 0)
    {
        byte_size = ((byte_size / alignment) + 1) * alignment;
    }

    void *random_values = aligned_alloc(alignment, byte_size);
    fill_random_data(random_values, dtype, size, min_range, max_range);

    Data *data = (Data *)malloc(sizeof(Data));
    data->values = random_values;
    data->size = size;
    data->dim = dim;
    data->shape = shape;
    data->dtype = dtype;

    total_data_allocated += sizeof(Data);
    add_data_ptr(data);

    return data;
}

/* Access an element in the Data object. */
void *get_data_element(Data *data, int *indices)
{
    int linear_index = calculate_index(indices, data->shape, data->dim);
    int dtype_size = get_data_size(data->dtype);

    // Cast the data values to the correct type and return the address of the element
    switch (data->dtype)
    {
    case FLOAT16:
    {
        float16 *values = (float16 *)data->values;
        return (void *)&values[linear_index];
    }
    case FLOAT32:
    {
        float32 *values = (float32 *)data->values;
        return (void *)&values[linear_index];
    }
    case FLOAT64:
    {
        float64 *values = (float64 *)data->values;
        return (void *)&values[linear_index];
    }
    default:
        printf("Unsupported dtype %s\n", get_data_type(data->dtype));
        return NULL;
    }
}

/* Set the value of an element in the Data object. */
void set_data_element(Data *data, int *indices, void *value)
{
    void *element_ptr = get_data_element(data, indices);

    // Set the value of the element
    switch (data->dtype)
    {
    case FLOAT16:
    {
        float16 *element = (float16 *)element_ptr;
        *element = *(float16 *)value;
        break;
    }
    case FLOAT32:
    {
        float32 *element = (float32 *)element_ptr;
        *element = *(float32 *)value;
        break;
    }
    case FLOAT64:
    {
        float64 *element = (float64 *)element_ptr;
        *element = *(float64 *)value;
        break;
    }
    default:
        printf("Failed to set element: Unsupported dtype %s\n", get_data_type(data->dtype));
        break;
    }
}

void fill_random_data(void *array, int dtype, int size, int min_range, int max_range)
{
    srand((unsigned int)time(NULL));
    switch (dtype)
    {
    case FLOAT16:
    {
        float16 *ptr = (float16 *)array;
        for (int i = 0; i < size; i++)
        {
            float random_value = min_range + ((float)rand() / (float)RAND_MAX) * (max_range - min_range);
            ptr[i] = float16_from_float(random_value);
        }
        break;
    }
    case FLOAT32:
    {
        float32 *ptr = (float32 *)array;
        for (int i = 0; i < size; i++)
        {
            ptr[i] = (float32)(min_range + ((float32)rand() / (float32)RAND_MAX) * (max_range - min_range));
        }
        break;
    }
    case FLOAT64:
    {
        float64 *ptr = (float64 *)array;
        for (int i = 0; i < size; i++)
        {
            ptr[i] = (float64)(min_range + ((float64)rand() / (float64)RAND_MAX) * (max_range - min_range));
        }
        break;
    }
    default:
        printf("Unsupported dtype for random fill %s\n", get_data_type(dtype));
        break;
    }
}
