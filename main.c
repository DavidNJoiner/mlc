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
	float weights[2];
	float bias;
} Parameters;

typedef struct {
	Data* data;
	float* gradient;
	int* shape;
	int dim;
	int stride;
} Tensor;

Data* createDataf(float* raw_data){
	(void)raw_data;
}

Tensor* createTensor(Data* data, int rows, int cols, int dim) {
	Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
	tensor->data = malloc(sizeof(*(tensor->data)) * rows * cols);
	tensor->gradient = (float*)calloc(rows * cols, sizeof(float));
	tensor->shape = (int*)malloc(sizeof(int) * 2);	
	tensor->stride = sizeof(data[0])*cols;
	tensor->shape[0] = rows;
	tensor->shape[1] = cols;
	tensor->dim = dim;
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

void flatten(Tensor* A){
	Data flattened_data[A->shape[0]*A->shape[1]];
	printf("Flattened tensor dimension = %zu",(size_t)(sizeof(flattened_data)/sizeof(A->data)));
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
	float data[4][2] = {{{0, 0, 2}, {0, 1, 3}, {1, 0, 2}, {1, 1, 8}},{{0, 0, 2}, {0, 1, 3}, {1, 0, 2}, {1, 1, 1}}};
	Data* dat = createDataf(data);
	Tensor* t2 = createTensor(dat, 4, 2, 2);
	Tensor* t1 = createTensor(dat, 4, 2, 2); 
	Tensor* res = createTensor(dat, 4, 2, 2);
    //mult(res,t2,t1);
	//add(res,t2);	
	//printTensor(res);
	//printf("Tensor size = %zu\n", sizeof(res)); 
	//	Parameters params = train(data, 4);

	//	printf("Trained weights:\n");
	//	printf("w1 = %f\n", params.weights[0]);
	//	printf("w2 = %f\n", params.weights[1]);
	//	printf("b = %f\n", params.bias);

	//	test(data, 4, params);

	return 0;
}

