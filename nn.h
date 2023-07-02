#ifndef OPS_H_ 
#define OPS_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tensor.h"
#include "data.h"
#include "dtype.h"

#define LEARNING_RATE 0.1
#define EPOCHS 1000

typedef struct {
    Parameters* params;
    float (*activation)(float); //Pointer to the a activation function.
} Neuron;

typedef struct {
    Neuron* neurons;
    int num_neurons;
} Layer;

typedef struct {
    Layer* layers;
    int num_layers;
} NN;

typedef struct {
    Tensor* w;
    Tensor* b;
} Parameters;

Parameters train(Data* data);
float sigmoid(float x);
float sigmoid_derivative(float x);

#endif //OPS_H