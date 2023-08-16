#include "config.h"
#include "mempool.h"
#include "tensor.h"
#include "nn.h"

#include "mempool.h"
#include <stdio.h>

void test_memory_pool()
{
    printf("Starting memory pool tests...\n");

    // 1. Initialize the memory pool
    printf("\nInitializing memory pool...\n");
    setup_pool(0, 1024); // Initialize a pool with 1024 bytes
    Pool_t *pool = fetch_pool();

    // 2. Allocate memory blocks
    printf("\nAllocating memory blocks...\n");
    MemoryBlock_t *block1 = create_block(pool);
    MemoryBlock_t *block2 = create_block(pool);
    MemoryBlock_t *block3 = create_block(pool);

    // 3. Display memory pool statistics
    printf("\nMemory pool statistics after allocations:\n");
    display_pool_stats(pool);

    // 4. Free some memory blocks
    printf("\nFreeing some memory blocks...\n");
    free_block(pool, block2);

    // 5. Display memory pool statistics again
    printf("\nMemory pool statistics after freeing:\n");
    display_pool_stats(pool);

    // 6. Destroy the memory pool
    printf("\nDestroying memory pool...\n");
    destroy_pool(pool);

    printf("Memory pool tests completed.\n");
}

int main()
{

    test_memory_pool();

    // getDevices(); Work in Progress
    ////Device *gpu = init_device(CUDA, 0);
    ////Device *cpu = init_device(CPU, -1);

    getDevices();

    // Initialize the global memory pool
    ////setup_tensor_pool(1);
    ////setup_global_data_ptr_array(1);

    // Create a tensor
    ////int shape[] = {16, 512};

    ////Data *data001 = random_data(8192, 0, 1, shape, 2, FLOAT32);
    // Data* data002 = random_data(8192, 0, 1, shape, 2, FLOAT32);

    // Tensor* t001 = tensor(data001, gpu, false);
    // Tensor* t002 = tensor(data002, gpu, false);
    ////Tensor *t003 = tensor(data001, cpu, false);
    // Tensor* t004 = tensor(data001, cpu, false);
    // Tensor* t004 = tensor(data002, cpu, false);
    // Tensor* gpures = zerosFrom(t002);
    // Tensor* cpures = zerosFrom(t004);

    // uint64_t s0 = nanos();
    // mul(gpures, t001, t002);
    // uint64_t e0 = nanos();
    // displayTensor(t9);

    // printf("\t \t \t \t CUDA Time: %f ms\n", (double)(e0 - s0) / 1000000.0);

    // uint64_t s1 = nanos();
    // mul(cpures, t003, t004);
    // uint64_t e1 = nanos();
    // displayTensor(t12);

    // printf("\t \t \t \t AVX Time: %f ms\n", (double)(e1 - s1) / 1000000.0);

    // Free the tensors
    ////printf("\nStarting cleaning...\n");
    ////free_all_tensors();

    // Free Pool
    ////destroy_tensor_pool();

    // Free the global memory pool
    ////free_device(gpu);
    ////free_device(cpu);

    return 0;
}