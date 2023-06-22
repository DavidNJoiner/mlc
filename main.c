#include "tensor.h"

#define LEARNING_RATE 0.1
#define EPOCHS 1000

int main() {

    float32 arr[4][3] =  {{2, 2, 2},
                        {3, 3, 3},
                        {4, 4, 4},
                        {5, 5, 5}};
                        
    float32 arr2[4][3] = {{2, 2, 2},
                        {3, 3, 3},
                        {4, 4, 4},
                        {5, 5, 5}};

    float64 arr3[2][4][3] = {{{0, 0, 5},
                            {0, 1, 5},
                            {1, 0, 1},
                            {1, 1, 0}},
                            {{0, 0, 5},
                            {0, 1, 5},
                            {1, 0, 1},
                            {1, 1, 0}}};

    int shape[] = {4, 3};
    int dim = 2;
    int shape1[] = {2, 4, 3};
    int dim1 = 3;

    Data* data = convertToData((void*)arr, shape, dim, FLOAT32);
    Data* data2 = convertToData((void*)arr2, shape, dim, FLOAT32);

    Tensor* t1 = tensor(data, false);
    Tensor* t2 = tensor(data2, false);
    Tensor* res = zerosFrom(t2);
    // Create a new Tensor from scratch
    //Tensor* res1 = createTensor(shape, 2, FLOAT32, false);
    //Tensor* t4 = tensor(data3, false);

    mult(res,t1,t2);
    printTensor(res);
    add(res,t2);
    printTensor(res);
    //add(t4, t4);
    //printTensor(t4);
  
    // Deallocate memory
    freeTensor(t1);
    freeTensor(t2);


    return 0;
}