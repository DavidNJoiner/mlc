#include "tensor.h"

// Setting requires_grad=True for a tensor means that the operations involving this tensor are tracked
// so that the gradient computations can be automatically done during backpropagation.

/*  -------------------------------------------------------*/
/*  tensor : create a new Tensor from a Data object.       */
/*  -------------------------------------------------------*/
// TODO : check if device is available on the system 
Tensor* tensor(Data* data, Device* device, bool requires_grad) {
    Tensor* new_tensor = (Tensor*)malloc(sizeof(Tensor));
    new_tensor->require_grad = requires_grad;
    if (requires_grad) {
        float32* gradient;
        if (device->type == CUDA) {
            cudaMalloc((void*)gradient, data->size * sizeof(float32));  
        } else {
            gradient = (float32*)calloc(data->size, sizeof(float32)); 
        }
        new_tensor->gradient = gradient;
    } else {
        new_tensor->gradient = NULL;
    }
    new_tensor->data = data;
    new_tensor->device = device;
    return new_tensor;
}
/*  -------------------------------------------------------*/
/*  createTensor : create a new Tensor from scratch.       */
/*  -------------------------------------------------------*/
// TODO : check if device is available on the system 
Tensor* createTensor(int* shape, int dim, int dtype, Device* device, bool requires_grad) {
    int size = 1;
    for (int i = 0; i < dim; i++) {
        size *= shape[i];
    }
    void* array;
    if (device->type == CUDA) {
        cudaMalloc(&array, size * GetDTypeSize(dtype));  
    } else {
        array = calloc(size, GetDTypeSize(dtype)); 
    }
    if (array == NULL) {
        printf("Memory allocation failed!\n");
        return NULL;
    }
    Data* data = MakeData(array, shape, dim, dtype);
    Tensor* t = tensor(data, device, requires_grad);
    return t;
}
/*  -------------------------------------------------------------------------------------*/
/*  zerosFrom : create a new Tensor filled with zeros from an existing Tensor(template). */
/*  -------------------------------------------------------------------------------------*/
Tensor* zerosFrom(Tensor* t) {
    Tensor* new_tensor = (Tensor*)malloc(sizeof(Tensor));
    Data* new_data = (Data*)malloc(sizeof(Data));

    new_data->shape = (int*)malloc(t->data->dim * sizeof(int));
    for (int i = 0; i < t->data->dim; i++) {
        new_data->shape[i] = t->data->shape[i];
    }
    new_data->dim = t->data->dim;
    new_data->size = t->data->size;
    new_data->dtype = t->data->dtype;

    if (t->device->type == CUDA) {
        cudaMalloc(&(new_data->values), new_data->size * GetDTypeSize(new_data->dtype));  
    } else {
        new_data->values = (float32 *)aligned_alloc(32, new_data->size * GetDTypeSize(new_data->dtype));  
    }

    new_tensor->data = new_data;
    new_tensor->device = t->device;
    new_tensor->require_grad = t->require_grad;

    if (t->gradient != NULL) {
        float32* gradient;
        if (t->device->type == CUDA) {
            cudaMalloc((void*)gradient, new_data->size * sizeof(float32));  
        } else {
            gradient = (float32*)calloc(new_data->size, sizeof(float32));  
        }
        new_tensor->gradient = gradient;
    } else {
        new_tensor->gradient = NULL;
    }

    return new_tensor;
}

Tensor* newFull(int* shape, int fill_value, int dtype, Device* device, bool requires_grad){
    
} 
/*  ---------------------------------------------------------------*/
/*  freeTensor : Releases the memory allocated for a given tensor. */
/*  ---------------------------------------------------------------*/
void freeTensor(Tensor* t) {
    if (t != NULL) {
        if (t->data != NULL) {
            if (t->data->values != NULL) {
                if (t->device->type == CUDA) {
                    cudaFree(t->data->values); 
                } else {
                    free(t->data->values);  
                }
                t->data->values = NULL;
            }
            free(t->data);
            t->data = NULL;
        }
        if (t->gradient != NULL) {
            if (t->device->type == CUDA) {
                cudaFree(t->gradient); 
            } else {
                free(t->gradient); 
            }
            t->gradient = NULL;
        }
        free(t);
        t = NULL;
    }
}
/*  ---------------------------------------------------------------*/
/*  shapesAreEqual : Check if two Tensors shapes are equals.       */
/*  ---------------------------------------------------------------*/
bool shapesAreEqual(Tensor* A, Tensor* B) {
    if (A->data->dim != B->data->dim) {
        printf("Dim mismatch in tensors!\n");
        return false;
    }

    for (int i = 0; i < A->data->dim; i++) {
        if (A->data->shape[i] != B->data->shape[i]) {
            printf("Shape mismatch in tensors! ");
            return false;
        }
    }

    return true;
}
/*  ---------------------------------------------------------------*/
/*  SameDevice : Check if n-Tensors are on the same device.        */
/*  ---------------------------------------------------------------*/
bool SameDevice(int num_tensors, ...){
    va_list args;
    va_start(args, num_tensors);

    Tensor* first_tensor = va_arg(args, Tensor*);
    Device* reference_device = first_tensor->device;
    
    for (int i = 1; i < num_tensors; i++){
        Tensor* tensor = va_arg(args, Tensor*);
        if(tensor->device != reference_device){
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
void mul(Tensor* dst, Tensor* A, Tensor* B) {

    if (!shapesAreEqual(A, B) || !shapesAreEqual(A, dst)) {
        return;
    }

    if(!SameDevice(3, dst, A, B)){
        return;
    }

    if (is_aligned(dst->data->values, 32) && is_aligned(A->data->values, 32) && is_aligned(B->data->values, 32)) {
        speed_mul_op(dst->data, A->data, B->data, dst->device);
    } else {
        printf("values are NOT 32-byte aligned.\n");
    }
}
/*  -----------------------------------------------------------------------------*/
/*  add : Add two Tensors A and B. Stores the result as a third Tensor dst   */
/*  -----------------------------------------------------------------------------*/
void add(Tensor* dst, Tensor* A) {

    if (!shapesAreEqual(A, dst)) {
        return;
    }

    if(!SameDevice(3, dst, A)){
        return;
    }

    if (is_aligned(dst->data->values, 32) && is_aligned(A->data->values, 32)) {
        speed_add_op(dst->data, A->data, dst->device);
    } else {
        printf("values are NOT 32-byte aligned.\n");
    }
}
/*  ---------------------------------------------------------------*/
/*   printTensor : print a Tensor to the console.                  */
/*  ---------------------------------------------------------------*/
void printTensor(Tensor* A) {

    if(A == NULL) {
        printf("Error: Null Tensor pointer passed to printTensor function.\n");
        return;
    }

    if(A->data == NULL) {
        printf("Error: Null Data pointer inside Tensor structure.\n");
        return;
    }

    if(A->data->shape == NULL) {
        printf("Error: Null Shape pointer inside Tensor structure.\n");
        return;
    }

    if (0 < A->data->dtype && A->data->dtype <= 16) {
        PrintData(A->data);
    }else {
        printf("Error: Invalid dtype.\n");
    }
}