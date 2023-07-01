#include "config.h"
#include "data.h"
#include "tensor.h"

int main() {

    int range[] = {0, 1};
    int shape[] = {8, 512};

    Data* data7 = randomData(4096, range, shape, 2, FLOAT16);
    Data* data8 = randomData(4096, range, shape, 2, FLOAT16);
    Data* data10 = randomData(4096, range, shape, 2, FLOAT32);
    Data* data11 = randomData(4096, range, shape, 2, FLOAT32);

    Device* deepc_gpu =  init_device(CUDA, 0);
    Device* deepc_cpu =  init_device(CPU, -1);

    Tensor* t7 = tensor(data7, deepc_gpu, false);
    Tensor* t8 = tensor(data8, deepc_gpu, false);
    Tensor* t9 = zerosFrom(t8);

    Tensor* t10 = tensor(data10, deepc_cpu, false);
    Tensor* t11 = tensor(data11, deepc_cpu, false);
    Tensor* t12 = zerosFrom(t11);

    uint64_t s0 = nanos();
    mul(t9, t7, t8);
    uint64_t e0 = nanos();
    //printTensor(t9);

    printf("\t \t \t CUDA Time: %f ms\n", (double)(e0 - s0) / 1000000.0);

    uint64_t s1 = nanos();
    mul(t12, t10, t11);
    uint64_t e1 = nanos();
    //printTensor(t9);

    printf("\t \t \t AVX Time: %f ms\n", (double)(e1 - s1) / 1000000.0);
    
    freeTensor(t7);
    freeTensor(t8);
    freeTensor(t9);
    freeTensor(t10);
    freeTensor(t11);
    freeTensor(t12);

    return 0;
}