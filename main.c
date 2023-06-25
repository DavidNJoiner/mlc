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

    //Data* data = convertToData((void*)arr, shape, dim, FLOAT32);
    //Data* data2 = convertToData((void*)arr2, shape, dim, FLOAT32);
    //Data* data3 = convertToData((void*)arr3, shape, dim, FLOAT32);

    //Data* data4 = convertToData((void*)arr4, shape1, dim1, FLOAT64);
    //Data* data5 = convertToData((void*)arr5, shape1, dim1, FLOAT64);
    //Data* data6 = convertToData((void*)arr6, shape1, dim1, FLOAT64);

    //Tensor* t1 = tensor(data, false);
    //Tensor* t2 = tensor(data2, false);
    //Tensor* t3 = tensor(data3, false);

    //Tensor* t4 = tensor(data4, false);
    //Tensor* t5 = tensor(data5, false);
    //Tensor* t6 = tensor(data6, false);
    //Tensor* t6 = zerosFrom(t5);

    // Create a new Tensor from scratch
    // Tensor* res1 = createTensor(shape, 2, FLOAT32, false);
    // Tensor* t4 = tensor(data3, false);

    //uint64_t start = nanos();
    //avx_mul(t6, t4, t5);
    //uint64_t end = nanos();
    //printTensor(t6);

    //printf("\t \t \t avx_mul Time: %f ms\n", (double)(end - start) / 1000000.0);

    //uint64_t start1 = nanos();
    //avx_add(t6, t4);
    //uint64_t end1 = nanos();
    //printTensor(t6);
    
    //printf("\t \t \t avx_add Time: %f ms\n", (double)(end1 - start1) / 1000000.0);
    
    //uint64_t start2 = nanos();
    //mult(t6, t4, t5);
    //uint64_t end2 = nanos();
    //printTensor(t6);

    //printf("\t \t \t Mult Time: %f ms\n", (double)(end2 - start2) / 1000000.0);

    //uint64_t start3 = nanos();
    //add(t6, t4);
    //uint64_t end3 = nanos();
    //printTensor(t6);

    //printf("\t \t \t Add Time: %f ms\n", (double)(end3 - start3) / 1000000.0);
    
    //freeTensor(t1);
    //freeTensor(t2);
    //freeTensor(t3);
    //freeTensor(t4);
    //freeTensor(t5);
    //freeTensor(t6);

    //  ******************************* //
    //   Large Matrices Multiplication  //
    //  ******************************* //

    int range[] = {0, 1};
    int shape2[] = {8, 512};

    Data* data7 = randomData(4096, range, shape2, 2, FLOAT32);
    Data* data8 = randomData(4096, range, shape2, 2, FLOAT32);

    Tensor* t7 = tensor(data7, false);
    Tensor* t8 = tensor(data8, false);
    Tensor* t9 = zerosFrom(t8);

    uint64_t s0 = nanos();
    mult(t9, t7, t8);
    uint64_t e0 = nanos();
    //printTensor(t9);

    printf("\t \t \t Mult Time: %f ms\n", (double)(e0 - s0) / 1000000.0);

    uint64_t s1 = nanos();
    avx_mul(t9, t7, t8);
    uint64_t e1 = nanos();
    //printTensor(t9);

    printf("\t \t \t avx_mul Time: %f ms\n", (double)(e1 - s1) / 1000000.0);
    
    freeTensor(t7);
    freeTensor(t8);
    freeTensor(t9);

    return 0;
}