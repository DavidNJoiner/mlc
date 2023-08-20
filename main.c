#include <stdio.h>
#include "config.h"
#include "core/mempool/mempool.h"
#include "tensor.h"
#include "nn.h"

void test_memory_pool(Pool_t *pool)
{
    printf("\nStarting memory pool tests...\n");

    printf("\nMemory pool statistics before allocations:\n");
    display_pool_stats(pool);

    printf("\nAllocating memory blocks...\n");
    MemoryBlock_ptr block1 = block_alloc(pool);
    MemoryBlock_ptr block2 = block_alloc(pool);
    MemoryBlock_ptr block3 = block_alloc(pool);

    printf("\nMemory pool statistics after allocations:\n");
    display_pool_stats(pool);

    printf("total_bytes_currently_allocated = %d\n", total_bytes_allocated);

    // 4. Free some memory blocks
    printf("\nFreeing some memory blocks...\n");
    free_block(pool, block2);

    // 5. Display memory pool statistics again
    printf("\nMemory pool statistics after freeing:\n");
    display_pool_stats(pool);

    printf("total_bytes_currently_allocated = %d\n", total_bytes_allocated);

    printf("\nMemory pool tests completed.\n");
}

void test_adding_subblocks()
{
    printf("Running test_adding_subblocks...\n");

    Pool_t *pool = fetch_pool();

    // Implement function to asign subblock automatically to the first free adjacent memory block.
    SubBlock_t *subblock1 = subblock_malloc(50, (MemoryBlock_t *)pool->m_next);
    SubBlock_t *subblock2 = subblock_malloc(100, (MemoryBlock_t *)pool->m_next);
    SubBlock_t *subblock3 = subblock_malloc(150, (MemoryBlock_t *)pool->m_next);

    assert(subblock1->m_size > 0);
    assert(subblock2->m_size > 0);
    assert(subblock3->m_size > 0);

    printf("test_adding_subblocks passed!\n");
}

void test_removing_last_subblock()
{
    printf("Running test_removing_last_subblock...\n");

    Pool_t *pool = fetch_pool();
    MemoryBlock_t *memblock = block_alloc(pool);

    DEEPC_VOID_POINTER subblock1 = subblock_malloc(50, memblock);
    DEEPC_VOID_POINTER subblock2 = subblock_malloc(100, memblock);
    DEEPC_VOID_POINTER subblock3 = subblock_malloc(150, memblock);

    remove_subblock(memblock, (SubBlock_t *)subblock3);

    // After removal, the last subblock should be marked as free (size 0)
    assert(((SubBlock_t *)subblock3)->m_size == 0);

    printf("test_removing_last_subblock passed!\n");
}

void test_buddy_system_merge()
{
    printf("Running test_buddy_system_merge...\n");

    Pool_t *pool = fetch_pool();
    MemoryBlock_t *memblock = block_alloc(pool);

    DEEPC_VOID_POINTER subblock1 = subblock_malloc(64, memblock);
    DEEPC_VOID_POINTER subblock2 = subblock_malloc(64, memblock);

    // Free the subblocks
    remove_subblock(memblock, (SubBlock_t *)subblock1);
    remove_subblock(memblock, (SubBlock_t *)subblock2);

    // Optimize the layout within the MemoryBlock after deletion
    optimize_layout(memblock);

    // After merging, we should have one larger free block instead of two smaller ones
    assert(count_blocks(6) == 1); // Assuming BLOCKSIZE is 64 and 6 is the order for 64 bytes

    printf("test_buddy_system_merge passed!\n");
}

int main()
{
    getDevices();

    printf("\nInitializing memory pool...\n");
    setup_pool(0, 4096); // Initialize a pool with 1024 bytes
    Pool_t *pool = fetch_pool();

    test_memory_pool(pool);
    // test_adding_subblocks();
    //  test_removing_last_subblock();
    //  test_buddy_system_merge();

    printf("\nDestroying memory pool...\n");
    destroy_pool(pool);

    display_table();

    // getDevices(); Work in Progress
    ////Device *gpu = init_device(CUDA, 0);
    ////Device *cpu = init_device(CPU, -1);

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