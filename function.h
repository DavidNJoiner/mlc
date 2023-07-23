#ifndef FUNCTION_H_ 
#define FUNCTION_H_

#include "device.h"
#include "tensor.h"

// Initialize a new Function example : Function* sin_function = InitFunction(device, tensors, Sin_Forward, Sin_Backward);

// needs_input_grad :
/*---------------------------------------------------------------------------------------------------------------------------
/* needs_input_grad is indeed a pointer to an array of boolean values corresponding to whether
/* each parent tensor requires a gradient. If the Function produces an output tensor that requires gradient computation,
/* it may need the gradients of its inputs for the Backward pass. needs_input_grad is constructed to keep track of which
/* inputs require gradients to optimize computation - if an input tensor doesn't require a gradient,
/* we don't need to compute it.
---------------------------------------------------------------------------------------------------------------------------*/

// requires_grad : 
/*---------------------------------------------------------------------------------------------------------------------------
/* Is a property of the Tensor class as well. Here it is set to True if any of the Function's parent tensors require a gradient.
/* In other words, if the function uses any tensor that needs a gradient, then the function itself will need to participate
/* in gradient computations.
---------------------------------------------------------------------------------------------------------------------------*/

// Forward declaration for Function
struct Function;

typedef struct  {
    Device* device;
    Data* parents;
    bool *needs_input_grad;
    bool requires_grad;
    Tensor* (*Forward)(struct Function *self, Data *args);
    void (*Backward)(struct Function *self, Data *args);
} Function;

Function* InitFunction(Device* device, Data* tensors, Tensor* (*ForwardFunc)(Function *self, Data *args), void (*BackwardFunc)(Function *self, Data *args));
Tensor *Forward(Function *self, Data *args);
void Backward(Function *self, Data *args);

#endif //FUNCTION_H_

