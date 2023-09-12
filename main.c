#include "core/config.h"
#include "core/mempool/mempool.h"
#include "core/deep_time.h"
#include "tensor.h"
#include "nn.h"

void test_memory_pool(Pool_t *pool)
{
    printf("\nStarting memory pool tests...\n");

    printf("\nMemory pool statistics before allocations:\n");
    display_pool_stats(pool);

    printf("\nAllocating memory blocks...\n");
    MemoryBlock_t *block1 = memblock_alloc(pool);
    MemoryBlock_t *block2 = memblock_alloc(pool);
    MemoryBlock_t *block3 = memblock_alloc(pool);

    // memblock_print(block2); SEGFAULT HERE

    printf("\nMemory pool statistics after allocations:\n");
    display_pool_stats(pool);

    printf("\nFreeing some memory blocks...\n");
    memblock_free(pool, block2);

    printf("\nMemory pool statistics after freeing:\n");
    display_pool_stats(pool);

    printf("\nMemory pool tests completed.\n");
}

void test_adding_subblocks()
{
    printf("Running test_adding_subblocks...\n");

    Pool_t *pool = fetch_pool();

    // Implement function to asign subblock automatically to the first free adjacent memory block.
    SubBlock_t *subblock1 = subblock_alloc(50, pool->m_memStart);
    SubBlock_t *subblock3 = subblock_alloc(150, pool->m_memStart);

    assert(subblock1->m_size > 0);
    assert(subblock3->m_size > 0);

    printf("test_adding_subblocks passed!\n");
    printf("\nMemory pool statistics after adding subblocks:\n");
    display_pool_stats(pool);

    _subblock_free_(pool->m_memStart, (SubBlock_t *)subblock3);
    _subblock_free_(pool->m_memStart, (SubBlock_t *)subblock1);

    printf("test_removing_subblocks passed!\n");
    printf("\nMemory pool statistics after removing subblocks:\n");
    display_pool_stats(pool);
}

void test_buddy_system_merge()
{
    printf("Running test_buddy_system_merge...\n");

    Pool_t *pool = fetch_pool();
    MemoryBlock_t *memblock = memblock_alloc(pool);

    DEEPC_VOID_POINTER subblock1 = subblock_alloc(64, memblock);
    DEEPC_VOID_POINTER subblock2 = subblock_alloc(64, memblock);

    // Free the subblocks
    _subblock_free_(memblock, (SubBlock_t *)subblock1);
    _subblock_free_(memblock, (SubBlock_t *)subblock2);

    // Optimize the layout within the MemoryBlock after deletion
    _subblock_coalescing_(memblock);

    // After merging, we should have one larger free block instead of two smaller ones
    assert(count_blocks(6) == 1); // Assuming BLOCKSIZE is 64 and 6 is the order for 64 bytes

    printf("test_buddy_system_merge passed!\n");
}

void test_float16()
{
    float16 a = float16_from_float(3.5f);
    float b = (a._h + sqrt(a._h));
    a._h += b;
    b += a._h;
    b = a._h + 7;
    printf("float32 b = %f, sizeof = %zu\n", b, sizeof(b));
    printf("float16 a = %d, sizeof = %zu\n", a._h, sizeof(a));
}

int main(int argc, char **argv)
{
    // getDevices();
    Device *gpu = init_device(CUDA, 0);
    Device *cpu = init_device(CPU, -1);

    printf("Initializing memory pool...\n");
    init_pool(0, 4096);
    Pool_t *pool = fetch_pool();

    test_memory_pool(pool);
    display_table();
    // test_adding_subblocks();
    //  test_removing_last_subblock();
    //  test_buddy_system_merge();

    printf("Destroying memory pool...\n");
    destroy_pool(pool);

    display_table();

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