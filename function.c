#include "function.h"

// Implementation of a Function struct and its interface to interact with Tensor objects. 
// The Function struct is key in handling autograd computations (automatic differentiation) 
// When a Function is applied to some Tensors, it keeps track of the operation and the input tensors
// (in the ctx context object), allowing for gradient computation during backpropagation.
// If the tensor requires_grad and no_grad mode is not active, the context is stored in ret._ctx,
// which can be used later for gradient computations.

Function* InitFunction(Device* device, Data* tensors, Tensor* (*ForwardFunc)(Function *self, Data *args), void (*BackwardFunc)(Function *self, Data *args)) {
    Function *self = (Function *)malloc(sizeof(Function));
    self->device = device;
    self->parents = tensors;

    self->needs_input_grad = (bool *)malloc(tensors->size * sizeof(bool));
    // Loop throught the tensors to fill the needs_input_grad array.
    for (int i = 0; i < tensors->size; i++) {
        Tensor *t = get_data_element(tensors, &i);
        self->needs_input_grad[i] = t->require_grad;
    }

    // If only one parent Tensor require_grad then the resulting Function will require it as well.
    for (int i = 0; i < tensors->size; i++) {
        if (self->needs_input_grad[i]) {
            self->requires_grad = true;
            break;
        }
    }

    // Assign the function pointers
    self->Forward = ForwardFunc;
    self->Backward = BackwardFunc;

    return self;
}


/*  ---------------------------------------------------------------*/
/*   Forward / Backward for autograd - Function member functions   */
/*  ---------------------------------------------------------------*/

Tensor* Forward(Function* self, Data* args) {
    printf("Forward not implemented for Function");
    exit(1);
}

void Backward(Function* self, Data* args) {
    printf("Backward not implemented for Function");
    exit(1);
}
