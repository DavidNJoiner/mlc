#include "config.h"
#include "mempool.h"
#include "tensor.h"
#include "nn.h"

int main() {

    // getDevices(); Work in Progress
    Device* gpu =  init_device(CUDA, 0);
    Device* cpu =  init_device(CPU, -1);

    cuda_version();

    // Initialize the global memory pool
    setupTensorPool(1);
    setupGlobalDataPtrArray(1);

    // Create a tensor
    int shape[] = {16, 512};

    Data* data001 = randomData(8192, 0, 1, shape, 2, FLOAT32);
    //Data* data002 = randomData(8192, 0, 1, shape, 2, FLOAT32);

    //Tensor* t001 = tensor(data001, gpu, false);
    //Tensor* t002 = tensor(data002, gpu, false);
    Tensor* t003 = tensor(data001, cpu, false);
    //Tensor* t004 = tensor(data001, cpu, false);
    //Tensor* t004 = tensor(data002, cpu, false);
    //Tensor* gpures = zerosFrom(t002);
    //Tensor* cpures = zerosFrom(t004);

    //uint64_t s0 = nanos();
    //mul(gpures, t001, t002);
    //uint64_t e0 = nanos();
    //displayTensor(t9);

    //printf("\t \t \t \t CUDA Time: %f ms\n", (double)(e0 - s0) / 1000000.0);

    //uint64_t s1 = nanos();
    //mul(cpures, t003, t004);
    //uint64_t e1 = nanos();
    //displayTensor(t12);

    //printf("\t \t \t \t AVX Time: %f ms\n", (double)(e1 - s1) / 1000000.0);
    
    // Free the tensors
    Pool* tensorPool = fetchPool(TENSOR);
    freeAllTensors(tensorPool);

    // Free Pool
    destroyPool(tensorPool);

    // Free the global memory pool
    free_device(gpu);
    free_device(cpu);

    return 0;
}