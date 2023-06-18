#include "tensor.h"

#define LEARNING_RATE 0.1
#define EPOCHS 1000

int main() {

    float32 arr[4][3] =  {{0, 0, 2},
                        {0, 1, 5},
                        {1, 0, 1},
                        {1, 1, 0}};
                        
    float32 arr2[4][3] = {{0, 0, 5},
                        {0, 1, 5},
                        {1, 0, 1},
                        {1, 1, 0}};

    int shape[] = {4, 3};
    int dim = 2;

    Data* data = convertToData((void*)arr, shape, dim, FLOAT32);
    Data* data2 = convertToData((void*)arr2, shape, dim, FLOAT32);

    

    Tensor* t1 = tensor(data, false);
    Tensor* t2 = tensor(data, false);
    Tensor* t3 = tensor(data2, false);
    Tensor* res = zerosFrom(t2);
    // Create a new Tensor from scratch
    Tensor* res1 = createTensor(shape, 2, FLOAT32, false);

    printTensor(t1);
    printTensor(t2);
    printTensor(res);
    printTensor(res1);

    mult(res,t1,t2);
    printTensor(res);
    add(res1,res);
    printTensor(res1);
  

    // Deallocate memory
    free(data);
    free(data2);
    free(t1);
    free(t2);
    free(t3);
    free(res1);
    free(res);

    return 0;
}