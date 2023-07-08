#include "config.h"
#include "memory_pool.h"
#include "tensor.h"
#include "nn.h"

int main() {

    // getDevices(); Work in Progress
    Device* gpu =  init_device(CUDA, 0);
    Device* cpu =  init_device(CPU, -1);

    cuda_version();

    // Initialize the global memory pool
    InitializeGlobalPool();

    // Create a tensor
    int shape[] = {16, 512};
    Data* data001 = RandomData(8192, 0, 1, shape, 2, FLOAT32);
    Data* data002 = RandomData(8192, 0, 1, shape, 2, FLOAT32);
    Tensor* t001 = tensor(data001, gpu, false);
    Tensor* t002 = tensor(data002, gpu, false);
    Tensor* t003 = tensor(data001, cpu, false);
    Tensor* t004 = tensor(data002, cpu, false);
    Tensor* res = zerosFrom(t002);

    uint64_t s0 = nanos();
    mul(res, t001, t002);
    uint64_t e0 = nanos();
    //printTensor(t9);

    printf("\t \t \t \t CUDA Time: %f ms\n", (double)(e0 - s0) / 1000000.0);

    uint64_t s1 = nanos();
    mul(res, t003, t004);
    uint64_t e1 = nanos();
    //printTensor(t12);

    printf("\t \t \t \t AVX Time: %f ms\n", (double)(e1 - s1) / 1000000.0);
    
    // Free the tensors
    Pool* tensorPool = GetPool(TENSOR);
    PoolDeepFree(tensorPool);

    // Free the global memory pool
    FreeGlobalPool();

    return 0;
}