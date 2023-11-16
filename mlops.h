#ifndef MLOPS_H_ 
#define MLOPS_H_

#include "function.h"

typedef struct {
    Function base; 
    Tensor* x; // Type will change to LazyBuffer
} Sin;

typedef struct {
    Function base;
    Tensor* ret; // Type will change to LazyBuffer
} Relu;

Tensor* SinForward(Sin* self, arr_t* args);
void SinBackward(Sin* self, arr_t* args);

Tensor* ReluForward(Relu* self, arr_t* args);
void ReluBackward(Relu* self, arr_t* args);



#endif //MLOPS_H_

#ifdef IMPLEMENTATION_MLOPS
#define IMPLEMENTATION_MLOPS

Sin* InitSin(Device* device, arr_t* tensors) {
    Sin* self = malloc(sizeof(Sin));
    self->base = *InitFunction(device, tensors, SinForward, SinBackward);
    return self;
}
Tensor* SinForward(Sin* self, Tensor* input) {
    Tensor* output = /* allocate and perform sin operation on input */;
    output->creator = (Function*)&(self->base); // The Sin function is the creator of the output Tensor
    return output;
}
void SinBackward(Sin* self, Tensor* grad_output) {
    /* Compute cosine of input */
    Tensor* cos_x = /* allocate and compute cos(self->base.parents->data) */;

    /* Compute product of cosine and incoming gradient */
    Tensor* grad_input = /* allocate and compute product of cos_x and grad_output->data */;

    /* If the input to the Sin function requires gradient, then we update its gradient.
     * Assuming the Sin function has only one input and is stored in self->base.parents.
     * TODO : Handle multiple input to the function.
     */
    if (self->base.parents->require_grad) {
        if (self->base.parents->gradient == NULL) {
            self->base.parents->gradient = grad_input->data;
        } else {
            /* add grad_input->data to self->base.parents->gradient */
        }
    }

    /* Clean up temporary Tensors */
    /* ... */
}


Relu* InitRelu(Device* device, arr_t* tensors) {
    Relu* self = malloc(sizeof(Relu));
    self->base = *InitFunction(device, tensors, ReluForward, ReluBackward);
    return self;
}

#endif //IMPLEMENTATION_MLOPS
