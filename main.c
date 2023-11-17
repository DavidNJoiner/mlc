#include "core/config.h"
#include "core/mempool/mempool.h"
#include "core/deep_time.h"
#include "tensor.h"
#include "nn.h"

void test_memory_pool(Pool_t *pool)
{
    printf("\nStarting memory pool tests...\n");

    printf("\nMemory pool statistics before allocations:\n");
    pool_print_stats(pool);

    printf("\nAllocating memory blocks...\n");
    MemBlock_t *block1 = memblock_alloc(pool);
    MemBlock_t *block2 = memblock_alloc(pool);
    MemBlock_t *block3 = memblock_alloc(pool);

    // memblock_print(block2); SEGFAULT HERE

    printf("\nMemory pool statistics after allocations:\n");
    pool_print_stats(pool);

    printf("\nFreeing some memory blocks...\n");
    memblock_free(pool, block2);

    printf("\nMemory pool statistics after freeing:\n");
    pool_print_stats(pool);

    printf("\nMemory pool tests completed.\n");
}

void test_adding_subblocks()
{
    printf("Running test_adding_subblocks...\n");

    Pool_t *pool = pool_get_from_index(0);

    // Implement function to asign subblock automatically to the first free adjacent memory block.
    SubBlock_t *subblock1 = subblock_alloc(50, pool->m_memStart);
    SubBlock_t *subblock3 = subblock_alloc(150, pool->m_memStart);

    assert(subblock1->m_size > 0);
    assert(subblock3->m_size > 0);

    printf("test_adding_subblocks passed!\n");
    printf("\nMemory pool statistics after adding subblocks:\n");
    pool_print_stats(pool);

    _subblock_free_(pool->m_memStart, (SubBlock_t *)subblock3);
    _subblock_free_(pool->m_memStart, (SubBlock_t *)subblock1);

    printf("test_removing_subblocks passed!\n");
    printf("\nMemory pool statistics after removing subblocks:\n");
    pool_print_stats(pool);
}

void test_buddy_system_merge()
{
    printf("Running test_buddy_system_merge...\n");

    Pool_t *pool = pool_get_from_index(0);
    MemBlock_t *memblock = memblock_alloc(pool);

    DEEPC_VOID_POINTER subblock1 = subblock_alloc(64, memblock);
    DEEPC_VOID_POINTER subblock2 = subblock_alloc(64, memblock);

    // Free the subblocks
    _subblock_free_(memblock, (SubBlock_t *)subblock1);
    _subblock_free_(memblock, (SubBlock_t *)subblock2);

    // Optimize the layout within the MemBlock after deletion
    _subblock_coalescing_(memblock);

    // After merging, we should have one larger free block instead of two smaller ones
    assert(pool_count_free_bytes(6) == 1); // Assuming BLOCKSIZE is 64 and 6 is the order for 64 bytes

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
    printf("\033[0;35mMAXIMUM UNSIGNED INT SIZE  %5u \033[0m\n", UINT_MAX);
    printf("\033[0;35mSIZE MemBlock_t            %5u \033[0m\n", BLOCKSIZE);

    test_float16();
    
    // getDevices();

    //Device *gpu = init_device(CUDA, 0);
    ////Device *cpu = init_device(CPU, -1);

    ////printf("Initializing memory pool...\n");
    ////pool_init(0, 4096);
    ////Pool_t *pool = pool_get_from_index(0);

    ////test_memory_pool(pool);
    ////display_table();
    // test_adding_subblocks();
    //  test_removing_last_subblock();
    //  test_buddy_system_merge();

    ////printf("Destroying memory pool...\n");
    ////pool_destroy(pool);

    ////display_table();

    // Initialize the global memory pool
    ////setup_tensor_pool(1);
    ////arr_init_global_ptr_count(1);

    // Create a tensor
    ////int shape[] = {16, 512};

    ////arr_t *data001 = arr_create_from_random(8192, 0, 1, shape, 2, FLOAT32);
    // arr_t* data002 = arr_create_from_random(8192, 0, 1, shape, 2, FLOAT32);

    // Tensor* t001 = tensor_from_array(data001, gpu, false);
    // Tensor* t002 = tensor_from_array(data002, gpu, false);
    ////Tensor *t003 = tensor_from_array(data001, cpu, false);
    // Tensor* t004 = tensor_from_array(data001, cpu, false);
    // Tensor* t004 = tensor_from_array(data002, cpu, false);
    // Tensor* gpures = tensor_zeros(t002);
    // Tensor* cpures = tensor_zeros(t004);

    // uint64_t s0 = nanos();
    // mul(gpures, t001, t002);
    // uint64_t e0 = nanos();
    // tensor_print(t9);

    // printf("\t \t \t \t CUDA Time: %f ms\n", (double)(e0 - s0) / 1000000.0);

    // uint64_t s1 = nanos();
    // mul(cpures, t003, t004);
    // uint64_t e1 = nanos();
    // tensor_print(t12);

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