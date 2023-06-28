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

    int range1[] = {0, 1};
    int shape1[] = {2, 4, 3};

    int range2[] = {0, 1};
    int shape2[] = {8, 512};

    Data* data7 = randomData(4096, range1, shape2, 2, FLOAT32);
    Data* data8 = randomData(4096, range2, shape2, 2, FLOAT32);

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