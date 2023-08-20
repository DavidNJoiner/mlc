#include "mempool.h"

static GlobalPool_t global_pool_instances = {0};

MemoryBlock_ptr memoryBlock_AddrFromIndex(const Pool_t *pool, uint32_t i)
{
    return pool->m_memStart + (i * pool->m_sizeOfEachBlock);
}

uint32_t count_blocks(uint32_t i)
{
    uint32_t count = 0;
    MemoryBlock_ptr block_ptr = memoryBlock_AddrFromIndex(&global_pool_instances.m_pools[0], i);
    uintptr_t *p = (uintptr_t *)&(*(block_ptr->freelist[i]));

    while (p != NULL)
    {
        count++;
        p = (uintptr_t *)*p;
    }
    return count;
}

uint32_t total_free()
{
    uint32_t i, bytecount = 0;

    for (i = 0; i <= MAX_ORDER; i++)
    {
        bytecount += count_blocks(i) * BLOCKSIZE * i;
    }
    return bytecount;
}

void display_pool_stats(Pool_t *pool)
{
    size_t poolSize = pool->m_numOfBlocks * pool->m_sizeOfEachBlock;

    double percentage_allocated_from_pool = poolSize == 0 ? 0 : (100.0 * total_bytes_allocated) / poolSize;

    printf("\t\033[34m[Info]\033[0m  Pool (Allocated / Total):          %10u / %lu\n", total_bytes_allocated, poolSize);
    printf("\t\033[34m[Info]\033[0m  Percentage Allocated:              %10.1lf%%\n", percentage_allocated_from_pool);
    printf("\t\033[34m[Info]\033[0m  Allocated memblock:                %10u\n", pool->m_numInitialized);
}

Pool_t *fetch_pool()
{
    if (!global_pool_instances.m_is_initialized)
    {
        setup_pool(0, 128);
    }
    else
    {
        return &global_pool_instances.m_pools[0];
    }
}

void setup_pool(uint8_t pool_instance_index, size_t pool_size)
{
    init_pool_(&global_pool_instances.m_pools[pool_instance_index], pool_size);
    global_pool_instances.m_is_initialized = true;
}

static void init_pool_(Pool_t *pool, const size_t poolSize)
{
    printf("[call] : init_pool_\n");
    init_table();
    static_assert(alignof(MemoryBlock_t) == DEEPC_SIZE_OF_VOID_POINTER, "\t[Error] MemoryBlock is not correctly aligned!\n");

    pool->m_numOfBlocks = (poolSize + sizeof(MemoryBlock_t) - 1) / sizeof(MemoryBlock_t);
    pool->m_sizeOfEachBlock = (uint32_t)sizeof(MemoryBlock_t);
    pool->m_memStart = (MemoryBlock_ptr)malloc(pool->m_sizeOfEachBlock * pool->m_numOfBlocks);
    pool->m_numFreeBlocks = pool->m_numOfBlocks;
    pool->m_next = pool->m_memStart;
    pool->m_numInitialized = 0;

    printf("[Debug] pool_size given to init pool = %zu\n", poolSize);
    printf("[Debug] num blocks = %d\n", pool->m_numOfBlocks);
    printf("[Debug] blocks unit size = %d\n", pool->m_sizeOfEachBlock);
    printf("[Debug] pool->m_numOfBlocks * pool->m_sizeOfEachBlock = %d\n", pool->m_numOfBlocks * pool->m_sizeOfEachBlock);
    printf("[Debug] pool start address : %p\n", (uintptr_t *)pool->m_memStart);
    printf("[Debug] pool end address : %p\n", (uintptr_t *)pool->m_memStart + pool->m_numOfBlocks * pool->m_sizeOfEachBlock);
    printf("[Debug] start address - end address : %d\n\n", (uint32_t)(pool->m_memStart - (pool->m_memStart + pool->m_numOfBlocks * pool->m_sizeOfEachBlock)));

    printf("\t\033[34m[Info]\033[0m  : initializing pool...        \n");
    printf("\t\033[34m[Info]\033[0m  : Pool size                %4u\n", pool->m_numOfBlocks * pool->m_sizeOfEachBlock);
    printf("\t\033[34m[Info]\033[0m  : Memblock available Qty   %4u\n", pool->m_numOfBlocks);
    printf("\t\033[34m[Info]\033[0m  : MemBlock unit size       %4u\n", pool->m_sizeOfEachBlock);
    printf("\t\033[34m[Info]\033[0m  : MemBlock initialized     %4u\n\n", pool->m_numInitialized);

    add_entry("init_pool", 2, (double)(sizeof(*pool->m_memStart)), 0.0);
    total_bytes_allocated += sizeof(*pool->m_memStart);
}

void destroy_pool(Pool_t *pool)
{
    printf("[call] : destroy_pool\n");
    void free_table();

    // free all memoryBlock present in the pool
    for (uint32_t i = 0; i <= pool->m_numInitialized; ++i)
    {
        MemoryBlock_ptr address_memblock_to_free = pool->m_memStart + (i * BLOCKSIZE);
        if (address_memblock_to_free->m_subblock_array) // Checking if a proper MemoryBlock is at the gathered address
        {
            free_block(pool, address_memblock_to_free);
            address_memblock_to_free = NULL;
        }
    }

    total_bytes_allocated -= sizeof(*pool->m_memStart);

    free((DEEPC_VOID_POINTER)pool->m_memStart);
    add_entry("destroy_pool", 2, 0.0, (double)(sizeof(*pool->m_memStart)));
    pool->m_memStart = (MemoryBlock_ptr)NULL;
}