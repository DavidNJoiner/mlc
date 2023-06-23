#include "tensor.h"


int main() {

    float32 arr[4][3] =  {{2, 2, 2},
                        {3, 3, 3},
                        {4, 4, 4},
                        {5, 5, 5}};
                        
    float32 arr2[4][3] = {{2, 2, 2},
                        {3, 3, 3},
                        {4, 4, 4},
                        {5, 5, 5}};

    float32 arr3[4][3] = {{0, 0, 0},
                        {0, 0, 0},
                        {0, 0, 0},
                        {0, 0, 0}};

    float64 arr4[2][4][3] = {{{0, 0, 5},
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
    Data* data3 = convertToData((void*)arr3, shape, dim, FLOAT32);

    Tensor* t1 = tensor(data, false);
    Tensor* t2 = tensor(data2, false);
    Tensor* t3 = tensor(data3, false);
    //Tensor* res = zerosFrom(t2);
    // Create a new Tensor from scratch
    // Tensor* res1 = createTensor(shape, 2, FLOAT32, false);
    // Tensor* t4 = tensor(data3, false);


    uint64_t start1 = nanos();
    fastmult(t3, t1, t2);
    uint64_t end1 = nanos();
    printTensor(t3);
    printf("------------------ Fastmult Time: %f s\n", (double)(end1 - start1) / 1000000000.0);
    
    uint64_t start2 = nanos();
    mult(t3, t1, t2);
    uint64_t end2 = nanos();
    printTensor(t3);
    printf("------------------ Mult Time: %f s\n", (double)(end2 - start2) / 1000000000.0);
    // add(t4, t4);
    // printTensor(t4);

    // Deallocate memory
    freeTensor(t1);
    freeTensor(t2);
    freeTensor(t3);


    return 0;
}