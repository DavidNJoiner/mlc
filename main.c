#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define LEARNING_RATE 0.1
#define EPOCHS 1000

typedef struct {
    float x1;
    float x2;
    float y;
} Data;

typedef struct {
    float weight[2];
    float bias;
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
    float weights[2]; //weights will be dynamically allocated based on the number of inputs to that neuron.
    float bias;
} Parameters;

typedef struct {
    Data* data;
    float* gradient;
    int* shape;
    int dim;
    int stride;
} Tensor;


void flatten(float* dat){
    // We assume the dat is always of type Data
    (void)dat;
}

Tensor* createTensor(Data* data, int rows, int cols, int dim) {
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    tensor->data = malloc(sizeof(*(tensor->data)) * rows * cols );
    tensor->gradient = (float*)calloc(rows * cols, sizeof(float));
    tensor->shape = (int*)malloc(sizeof(int) * dim);	
    tensor->stride = sizeof(sizeof(float)*cols); //Assuming the datatype of the dataset is float
    tensor->shape[0] = rows;
    tensor->shape[1] = cols;
    tensor->shape[2] = dim;
    tensor->data = data;
    return tensor;
}

void ones(Tensor* A){
    for (int i = 0; i < A->shape[0]; ++i) {
        for (int j = 0; j < A->shape[1]; ++j) {
            int index = i * A->shape[1] + j;
            A->data[index].x1 = 1;
            A->data[index].x2 = 1;
            A->data[index].y = 1;
        }
    }

}

void printTensor(Tensor* A){
    for (int i = 0; i < A->shape[0]; ++i) {
        for (int j = 0; j < A->shape[1]; ++j) {
            int index = i * A->shape[1] + j;
            printf("x1 = %2f", A->data[index].x1);
            printf(" x2 = %2f", A->data[index].x2);
            printf(" y = %2f", A->data[index].y);
            printf("\n");
        }
    }

}

void mult(Tensor* dst, Tensor* A, Tensor* B) {
    // Assuming A, B and dst are all of the same dimension
    for (int i = 0; i < A->shape[0]; ++i) {
        for (int j = 0; j < A->shape[1]; ++j) {
            int index = i * A->shape[1] + j;
            dst->data[index].x1 = A->data[index].x1 * B->data[index].x1;
            dst->data[index].x2 = A->data[index].x2 * B->data[index].x2;
            dst->data[index].y = A->data[index].y * B->data[index].y;
        }
    }
}

void add(Tensor* dst, Tensor* A) {
    for (int i = 0; i < A->shape[0]; ++i) {
        for (int j = 0; j < A->shape[1]; ++j) {
            int index = i * A->shape[1] + j;
            dst->data[index].x1 = A->data[index].x1 +  dst->data[index].x1;
            dst->data[index].x2 = A->data[index].x2 +  dst->data[index].x2;
            dst->data[index].y = A->data[index].y +  dst->data[index].y;
        }
    }
}

float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

float sigmoid_derivative(float x) {
    return x * (1 - x);
}

Parameters train(Data *data, int size) {
    Parameters params;
    params.weights[0] = (float)rand() / (float)RAND_MAX;
    params.weights[1] = (float)rand() / (float)RAND_MAX;
    params.bias = (float)rand() / (float)RAND_MAX;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        for (int i = 0; i < size; ++i) {
            float z = params.weights[0] * data[i].x1 + params.weights[1] * data[i].x2 + params.bias;
            float prediction = sigmoid(z);

            float error = data[i].y - prediction;

            params.weights[0] += LEARNING_RATE * error * sigmoid_derivative(prediction) * data[i].x1;
            params.weights[1] += LEARNING_RATE * error * sigmoid_derivative(prediction) * data[i].x2;
            params.bias += LEARNING_RATE * error * sigmoid_derivative(prediction);
        }
    }

    return params;
}

void test(Data *data, int size, Parameters params) {
    printf("Testing:\n");
    for (int i = 0; i < size; ++i) {
        float z = params.weights[0] * data[i].x1 + params.weights[1] * data[i].x2 + params.bias;
        float prediction = sigmoid(z);
        printf("Input: %f, %f. Output: %d\n", data[i].x1, data[i].x2, prediction > 0.5 ? 1 : 0);
    }
}

int main() {
    Data  dat[4] = {{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}};
    Tensor* t2 = createTensor(dat, 4, 1, 2);//Here dat decay into a pointer, which is what createTensor expect.
    Tensor* t1 = createTensor(dat, 4, 1, 2); 
    Tensor* res = createTensor(dat, 4, 1, 2);
    mult(res,t2,t1);
    add(res,t2);	
    printTensor(res);

    
    Data  dat2[2][4] = {{{1, 2, 3}, {0, 6, 1}, {2, 0, 8}, {1, 4, 0}}, {{0, 2, 3}, {0, 6, 1}, {2, 0, 8}, {1, 4, 0}}};
    Data* data_p = &dat2[0][0];//Flatten the Data array

    Tensor* t3 = createTensor(data_p, 4, 2, 2);
    Tensor* t4 = createTensor(data_p, 4, 2, 2); 
    Tensor* res1 = createTensor(data_p, 4, 2, 2);
    mult(res1,t4,t3);
    add(res1,t4);	
    printTensor(res1);
    //printf("Tensor size = %zu\n", sizeof(res)); 
    //	Parameters params = train(data, 4);

    //	printf("Trained weights:\n");
    //	printf("w1 = %f\n", params.weights[0]);
    //	printf("w3 = %f\n", params.weights[1]);
    //	printf("b = %f\n", params.bias);

    //	test(data, 4, params);

    return 0;
}

