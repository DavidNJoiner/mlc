#include "nn.h"

float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

float sigmoid_derivative(float x) {
    return x * (1 - x);
}

Parameters train(Model* model, Tensor* X_train, Tensor* Y_train, Optim* optim, int steps, int BS, LossF* loss_function) {
    Parameters params;
    int tw_size = params.w->data->size = data->size;
    params.w->data->dtype = data->dtype;
    // Randomly initialize weights and biases.
    for (uint32_t i = 0; i < tw_size; i++) {
        params.w[i] = ()rand() / ()RAND_MAX;
        params.b[i] = ()rand() / ()RAND_MAX;
    }

    

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        for (uint32_t i = 0; i < size; ++i) {
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

/* int test() {
    //arr_t  dat[4] = {{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}};
    //Tensor* t2 = tensor_from_scratch(dat, 4, 1, 2);//Here dat decay into a pointer, which is what tensor_from_scratch expect.
    //Tensor* t1 = tensor_from_scratch(dat, 4, 1, 2); 
    //Tensor* res = tensor_from_scratch(dat, 4, 1, 2);
    //mult(res,t2,t1);
    //add(res,t2);	
    //tensor_print(res);

    arr_t  dat2[2][4] = {{{1.f, 2.f, 3.f}, {0.f, 6.f, 1.f}, {2.f, 0.f, 8.f}, {1.f, 4.f, 0.f}}, {{0.f, 2.f, 3.f}, {0.f, 6.f, 1.f}, {2.f, 0.f, 8.f}, {1.f, 4.f, 0.f}}};
    arr_t* data_p = &dat2[0][0];//Flatten the arr_t array

    Tensor* t3 = tensor_from_scratch(data_p, 4, 2, 2);
    Tensor* t4 = tensor_from_scratch(data_p, 4, 2, 2); 
    Tensor* res1 = tensor_from_scratch(data_p, 4, 2, 2);
    mult(res1,t4,t3);
    add(res1,t4);	
    tensor_print(res1);
    //printf("Tensor size = %zu\n", sizeof(res)); 
    //	Parameters params = train(data, 4);

    //	printf("Trained weights:\n");
    //	printf("w1 = %f\n", params.weights[0]);
    //	printf("w3 = %f\n", params.weights[1]);
    //	printf("b = %f\n", params.bias);

    //	test(data, 4, params);

    return 0;
} */

