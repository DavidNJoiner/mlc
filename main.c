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
    
    float64 arr5[2][4][3] = {{{0, 0, 5},
                            {0, 1, 5},
                            {1, 0, 1},
                            {1, 1, 0}},
                            {{0, 0, 5},
                            {0, 1, 5},
                            {1, 0, 1},
                            {1, 1, 0}}};
    
    float64 arr6[2][4][3] = {{{0, 0, 0},
                            {0, 0, 0},
                            {0, 0, 0},
                            {0, 0, 0}},
                            {{0, 0, 0},
                            {0, 0, 0},
                            {0, 0, 0},
                            {0, 0, 0}}};

    int shape[] = {4, 3};
    int dim = 2;
    int shape1[] = {2, 4, 3};
    int dim1 = 3;

    Data* data = convertToData((void*)arr, shape, dim, FLOAT32);
    Data* data2 = convertToData((void*)arr2, shape, dim, FLOAT32);
    Data* data3 = convertToData((void*)arr3, shape, dim, FLOAT32);

    Data* data4 = convertToData((void*)arr4, shape1, dim1, FLOAT64);
    Data* data5 = convertToData((void*)arr5, shape1, dim1, FLOAT64);
    Data* data6 = convertToData((void*)arr6, shape1, dim1, FLOAT64);

    Tensor* t1 = tensor(data, false);
    Tensor* t2 = tensor(data2, false);
    Tensor* t3 = tensor(data3, false);

    Tensor* t4 = tensor(data4, false);
    Tensor* t5 = tensor(data5, false);
    Tensor* t6 = tensor(data6, false);

    // Create a new Tensor from scratch
    // Tensor* res1 = createTensor(shape, 2, FLOAT32, false);
    // Tensor* t4 = tensor(data3, false);

    fastmult(t6, t4, t5);
    printTensor(t6);
    fastadd(t6, t4);
    printTensor(t6);
    
    //uint64_t start2 = nanos();
    //mult(res, t1, t2);
    //uint64_t end2 = nanos();
    //printTensor(res);
    //printf("\t \t \t Mult Time: %f s\n \n", (double)(end2 - start2) / 1000000000.0);
    // add(t4, t4);
    // printTensor(t4);

    // Deallocate memory
    freeTensor(t1);
    freeTensor(t2);
    freeTensor(t3);
    freeTensor(t4);
    freeTensor(t5);
    freeTensor(t6);


    return 0;
}