#ifndef NN_H_ 
#define NN_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tensor.h"
#include "data/dataset.h"
#include "dtype.h"
#include "function.h"

#define LEARNING_RATE 0.1
#define EPOCHS 1000

typedef struct {
    Tensor* w;
    Tensor* b;
} Parameters;

typedef struct {
    Parameters* params;
    float (*Function)(float); //Pointer to the a activation function.
} Neuron;

typedef struct {
    Neuron* neurons;
    int num_neurons;
} Layer;

typedef struct {
    Layer* layers;
    int num_layers;
} NeuralNet;

Parameters train(Data* data);
float sigmoid(float x);
float sigmoid_derivative(float x);

#endif //NN_H