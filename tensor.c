#include "tensor.h"
#include "core/mempool/mempool.h"

// Setting requires_grad=True for a tensor means that the operations involving this tensor are tracked
// so that the gradient computations can be automatically done during backpropagation.


//Create a Tensor object from a arr_t object.
Tensor *tensor_from_array(arr_t *data, Device *device, bool requires_grad)
{
    // TODO : check if device is available on the system
    // Use the custom memory allocator to allocate memory for the new tensor
    Pool_t *Pool = pool_get_from_index(0);
    Tensor *new_tensor = (Tensor *)memblock_alloc(Pool);

    set_require_grad(new_tensor, requires_grad);
    if (requires_grad)
    {
        if (device->type == CUDA)
        {
            cudaMalloc((void **)&(new_tensor->gradient), data->size * sizeof(float32));
        }
        else
        {
            new_tensor->gradient = (float32 *)calloc(data->size, sizeof(float32));
        }
    }
    else
    {
        new_tensor->gradient = NULL;
    }

    new_tensor->data = data;
    new_tensor->device = device;

    printf("\t[DEBUG] Tensor created at address %p\n", (void *)new_tensor);

    return new_tensor;
}


// Create a new Tensor from scratch.
Tensor *tensor_from_scratch(int *shape, int dim, int dtype, Device *device, bool requires_grad)
{
    int size = 1;
    for (uint32_t i = 0; i < dim; i++)
    {
        size *= shape[i];
    }
    void *array;
    if (device->type == CUDA)
    {
        cudaMalloc(&array, size * get_data_size(dtype));
    }
    else
    {
        array = calloc(size, get_data_size(dtype));
    }
    if (array == NULL)
    {
        printf("Memory allocation failed!\n");
        return NULL;
    }
    arr_t *data = arr_create_from_array(array, shape, dim, dtype); // check if Array functions handle cuda memory
    Pool_t *Pool = pool_get_from_index(0);
    Tensor *t = (Tensor *)memblock_alloc(Pool);
    t->data = data;
    t->device = device;
    set_require_grad(t, (int)requires_grad);
    return t;
}

// Create a Tensor filled with zeros from an existing Tensor(template tensor).
Tensor *tensor_zeros(Tensor *t)
{

    Pool_t *Pool = pool_get_from_index(0);

    Tensor *new_tensor = (Tensor *)memblock_alloc(Pool);
    arr_t *new_data = (arr_t *)memblock_alloc(Pool);

    new_data->shape = (int *)malloc(t->data->dim * sizeof(int));
    if (new_data->shape == NULL)
    {
        printf("Error: Failed to allocate memory for new_data->shape\n");
        exit(1);
    }
    for (uint32_t i = 0; i < t->data->dim; i++)
    {
        new_data->shape[i] = t->data->shape[i];
    }
    new_data->dim = t->data->dim;
    new_data->size = t->data->size;
    new_data->dtype = t->data->dtype;

    if (t->device->type == CUDA)
    {
        if (cudaMalloc(&(new_data->values), new_data->size * get_data_size(new_data->dtype)) != cudaSuccess)
        {
            printf("Error: Failed to allocate GPU memory for new_data->values\n");
            exit(1);
        }
    }
    else
    {
        new_data->values = (float32 *)memory_malloc_aligned(32, new_data->size * get_data_size(new_data->dtype));
        if (new_data->values == NULL)
        {
            printf("Error: Failed to allocate memory for new_data->values\n");
            exit(1);
        }
    }

    new_tensor->data = new_data;
    new_tensor->device = t->device;
    set_require_grad(new_tensor, get_require_grad(t));

    if (t->gradient != NULL)
    {
        float32 *gradient;
        if (t->device->type == CUDA)
        {
            if (cudaMalloc((void **)&gradient, new_data->size * sizeof(float32)) != cudaSuccess)
            {
                printf("Error: Failed to allocate GPU memory for gradient\n");
                exit(1);
            }
        }
        else
        {
            gradient = (float32 *)calloc(new_data->size, sizeof(float32));
            if (gradient == NULL)
            {
                printf("Error: Failed to allocate memory for gradient\n");
                exit(1);
            }
        }
        new_tensor->gradient = gradient;
    }
    else
    {
        new_tensor->gradient = NULL;
    }

    return new_tensor;
}

Tensor *tensor_ones(int *shape, int fill_value, int dtype, Device *device, bool requires_grad)
{
    // TODO: Implement this function
    return NULL;
}

// Check if two Tensors shapes are equals.
bool shapesAreEqual(Tensor *A, Tensor *B)
{
    if (A->data->dim != B->data->dim)
    {
        printf("Dim mismatch in tensors!\n");
        return false;
    }

    for (uint32_t i = 0; i < A->data->dim; i++)
    {
        if (A->data->shape[i] != B->data->shape[i])
        {
            printf("Shape mismatch in tensors! ");
            return false;
        }
    }

    return true;
}

// Check if n-Tensors are on the same device.
bool sameDevice(int num_tensors, ...)
{
    va_list args;
    va_start(args, num_tensors);

    int mismatches = 0;
    Tensor *first_tensor = va_arg(args, Tensor *);
    Device *reference_device = first_tensor->device;

    for (int i = 1; i < num_tensors; i++)
    {
        Tensor *tensor = va_arg(args, Tensor *);
        if (tensor->device != reference_device)
        {
            va_end(args);
            mismatches += 1;
            return false;
        }
    }

    printf("[info] Device mismatch : %d\n", mismatches);
    va_end(args);
    return true;
}

// Multiply two Tensors A and B. Stores the result as a third Tensor dst
void mul(Tensor *dst, Tensor *A, Tensor *B)
{

    if (!shapesAreEqual(A, B) || !shapesAreEqual(A, dst))
    {
        return;
    }

    if (!sameDevice(3, dst, A, B))
    {
        return;
    }

    if (is_tensor_aligned(dst->data->values, 32) && is_tensor_aligned(A->data->values, 32) && is_tensor_aligned(B->data->values, 32))
    {
        speed_mul_op(dst->data, A->data, B->data, dst->device);
    }
    else
    {
        printf("values are NOT 32-byte aligned.\n");
    }
}

// Add two Tensors A and B. Stores the result as a third Tensor dst
void add(Tensor *dst, Tensor *A)
{

    if (!shapesAreEqual(A, dst))
    {
        return;
    }

    if (!sameDevice(3, dst, A))
    {
        return;
    }

    if (is_tensor_aligned(dst->data->values, 32) && is_tensor_aligned(A->data->values, 32))
    {
        speed_add_op(dst->data, A->data, dst->device);
    }
    else
    {
        printf("values are NOT 32-byte aligned.\n");
    }
}

// Print a Tensor to the console.
void tensor_print(Tensor *A)
{
    // TODO : finish writing this function
    if (A == NULL)
    {
        printf("Error: Null Tensor pointer passed to tensor_print function.\n");
        return;
    }

    if (A->data == NULL)
    {
        printf("Error: Null Array pointer inside Tensor structure.\n");
        return;
    }

    if (A->data->shape == NULL)
    {
        printf("Error: Null Shape pointer inside Tensor structure.\n");
        return;
    }

    if (0 < A->data->dtype && A->data->dtype <= 16)
    {
        arr_print(A->data);
    }
    else
    {
        printf("Error: Invalid dtype.\n");
    }
}


// Memory Alignement Check.
bool is_tensor_aligned(void *ptr, size_t alignment)
{
    return ((uintptr_t)ptr % alignment) == 0;
}

// Set and Get Tensor gradient.
void set_require_grad(Tensor *tensor, int bit_flag)
{
    uint32_t *int_ptr = (uint32_t *)&tensor->gradient;
    *int_ptr |= (bit_flag << 31);
}
bool get_require_grad(Tensor *tensor)
{
    uint32_t *int_ptr = (uint32_t *)&tensor->gradient;
    return (*int_ptr >> 31) & 1;
}
