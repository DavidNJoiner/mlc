#include "tensor.h"
#include "core/mempool/mempool.h"

// Setting requires_grad=True for a tensor means that the operations involving this tensor are tracked
// so that the gradient computations can be automatically done during backpropagation.

/*  -------------------------------------------------------*/
/*  tensor : create a new Tensor from a Data object.       */
/*  -------------------------------------------------------*/
// TODO : check if device is available on the system
Tensor *tensor(Data *data, Device *device, bool requires_grad)
{

    // Use the custom memory allocator to allocate memory for the new tensor
    Pool_t *Pool = fetch_pool();
    Tensor *new_tensor = (Tensor *)block_alloc(Pool);

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

/*  -------------------------------------------------------*/
/*  create_tensor : create a new Tensor from scratch.       */
/*  -------------------------------------------------------*/
Tensor *create_tensor(int *shape, int dim, int dtype, Device *device, bool requires_grad)
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
    Data *data = create_data(array, shape, dim, dtype); // check if Data functions handle cuda memory
    Pool_t *Pool = fetch_pool();
    Tensor *t = (Tensor *)block_alloc(Pool);
    t->data = data;
    t->device = device;
    set_require_grad(t, (int)requires_grad);
    return t;
}
/*  -------------------------------------------------------------------------------------/
  zerosFrom : create a new Tensor filled with zeros from an existing Tensor(template).
/  -------------------------------------------------------------------------------------*/
Tensor *zerosFrom(Tensor *t)
{

    Pool_t *Pool = fetch_pool();

    Tensor *new_tensor = (Tensor *)block_alloc(Pool);
    Data *new_data = (Data *)block_alloc(Pool);

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
        new_data->values = (float32 *)aligned_alloc(32, new_data->size * get_data_size(new_data->dtype));
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

Tensor *newFull(int *shape, int fill_value, int dtype, Device *device, bool requires_grad)
{
    // TODO: Implement this function
    return NULL;
}
/*  ---------------------------------------------------------------*/
/*  shapesAreEqual : Check if two Tensors shapes are equals.       */
/*  ---------------------------------------------------------------*/
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
/*  ---------------------------------------------------------------*/
/*  sameDevice : Check if n-Tensors are on the same device.        */
/*  ---------------------------------------------------------------*/
bool sameDevice(int num_tensors, ...)
{
    va_list args;
    va_start(args, num_tensors);

    Tensor *first_tensor = va_arg(args, Tensor *);
    Device *reference_device = first_tensor->device;

    for (int i = 1; i < num_tensors; i++)
    {
        Tensor *tensor = va_arg(args, Tensor *);
        if (tensor->device != reference_device)
        {
            va_end(args);
            printf("Device mismatch.\n");
            return false;
        }
    }

    va_end(args);
    return true;
}
/*  --------------------------------------------------------------------------------*/
/*  mul : Multiply two Tensors A and B. Stores the result as a third Tensor dst */
/*  --------------------------------------------------------------------------------*/
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

    if (is_aligned(dst->data->values, 32) && is_aligned(A->data->values, 32) && is_aligned(B->data->values, 32))
    {
        speed_mul_op(dst->data, A->data, B->data, dst->device);
    }
    else
    {
        printf("values are NOT 32-byte aligned.\n");
    }
}
/*  -----------------------------------------------------------------------------*/
/*  add : Add two Tensors A and B. Stores the result as a third Tensor dst   */
/*  -----------------------------------------------------------------------------*/
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

    if (is_aligned(dst->data->values, 32) && is_aligned(A->data->values, 32))
    {
        speed_add_op(dst->data, A->data, dst->device);
    }
    else
    {
        printf("values are NOT 32-byte aligned.\n");
    }
}
/*  ---------------------------------------------------------------*/
/*   displayTensor : print a Tensor to the console.                  */
/*  ---------------------------------------------------------------*/
void displayTensor(Tensor *A)
{

    if (A == NULL)
    {
        printf("Error: Null Tensor pointer passed to displayTensor function.\n");
        return;
    }

    if (A->data == NULL)
    {
        printf("Error: Null Data pointer inside Tensor structure.\n");
        return;
    }

    if (A->data->shape == NULL)
    {
        printf("Error: Null Shape pointer inside Tensor structure.\n");
        return;
    }

    if (0 < A->data->dtype && A->data->dtype <= 16)
    {
        display_data(A->data);
    }
    else
    {
        printf("Error: Invalid dtype.\n");
    }
}

/*  -------------------------------------------------------*/
/*  Memory Alignement Check                                */
/*  -------------------------------------------------------*/
bool is_aligned(void *ptr, size_t alignment)
{
    return ((uintptr_t)ptr % alignment) == 0;
}
void set_require_grad(Tensor *tensor, int bit_flag)
{
    tensor->gradient |= (bit_flag << 31);
}
bool get_require_grad(Tensor *tensor)
{
    return (tensor->gradient >> 31) & 1;
}