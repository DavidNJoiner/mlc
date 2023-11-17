#include "mempool.h"

static GlobalPool_t global_pool_instances = {0};

/*-------------------------------------------------------*/
/*        POOL : Memory management functions             */
/*-------------------------------------------------------*/

/* Retrieves a pool pointer based on its index. */
Pool_t* pool_get_from_index(int pool_index)
{
    return &global_pool_instances.m_pools[pool_index];
}

uint32_t pool_count_free_bytes(int pool_index)
{
    uint32_t bytecount = 0;
    Pool_t *pool_ptr = &global_pool_instances.m_pools[pool_index];

    bytecount = pool_ptr->m_numFreeBlocks * pool_ptr->m_sizeOfEachBlock;

    // TODO : Account for free bytes within each memblock

    return bytecount;
}

void pool_init(uint8_t pool_instance_index, size_t pool_size)
{
    // static_assert(sizeof(MemBlock_t) == 1120, "\t[Error] MemBlock is not 1024 bytes!\n");
    typedef char check_memblock_alignment[(alignof(MemBlock_t) == DEEPC_SIZE_OF_VOID_POINTER) ? 1 : -1];

    Pool_t* pool = &global_pool_instances.m_pools[pool_instance_index];

    pool->m_numOfBlocks = (pool_size + BLOCKSIZE - 1) / BLOCKSIZE;
    pool->m_sizeOfEachBlock = (uint32_t)sizeof(MemBlock_t);
    pool->m_memStart = (MemBlock_t *)malloc(pool->m_sizeOfEachBlock * pool->m_numOfBlocks); // (MemBlock_t*)malloc(DEEPC_SIZE_OF_VOID_POINTER);
    pool->m_numFreeBlocks = pool->m_numOfBlocks;
    pool->m_next = pool->m_memStart;
    pool->m_numInitialized = 0;
    global_pool_instances.m_is_initialized = true;

    pool_init_debug(pool, pool_size);
    
    init_table(); // Table should be init somewhere else.
    
    // Update globals
    increase_total_bytes_allocated(sizeof(pool->m_memStart));
    add_entry("pool_init", 2, (double)(sizeof(pool->m_memStart)), 0.0);
}

void pool_destroy(Pool_t *pool)
{
    void free_table(); // Table should be freed somewhere else.

    // free all memoryBlock present in the pool
    for (uint32_t i = 0; i <= pool->m_numInitialized; ++i)
    {
        MemBlock_t *address_memblock_to_free = (MemBlock_t *)((uintptr_t)pool->m_memStart + i * BLOCKSIZE);

        block_free(pool, address_memblock_to_free);
        address_memblock_to_free = NULL;
    }

    // Update globals
    decrease_total_bytes_allocated(sizeof(pool->m_memStart));
    free((DEEPC_VOID_POINTER)pool->m_memStart);
    add_entry("pool_destroy", 2, 0.00, (double)(sizeof(pool->m_memStart)));
    pool->m_memStart = NULL;
}

/*-------------------------------------------------------*/
/*       MEMBLOCK : Memory management functions          */
/*-------------------------------------------------------*/

/* Returns a pointer to a memory block at a given index within a pool. */
MemBlock_t *memoryBlock_get_ptr_from_index(const Pool_t *pool, int i)
{
    return pool->m_memStart + (i * pool->m_sizeOfEachBlock);
}

/* Counts the total size of free subblocks within a memory block. */
uint32_t memblock_count_free_subblocks(MemBlock_t* memblock_ptr)
{
    uint32_t i, bytecount = 0;
    for (i; i <= MAX_ORDER; i++)
    {
        SubBlock_t *subblock = memblock_ptr->freelist[i];
        bytecount += subblock->m_size; // Automatic cast from size_t to uint32_t. Might not be the best approach. 
    }
    return bytecount;
}


/*-------------------------------------------------------*/
/*              POOL : Debug functions                   */
/*-------------------------------------------------------*/

void pool_print_stats(Pool_t *pool)
{
    size_t poolSize = pool->m_numOfBlocks * pool->m_sizeOfEachBlock;
    int allocated_bytes = get_total_bytes_allocated();
    printf("\t\033[0;32m[Debug]\033[0m Display_pool_stats : \nTotal bytes allocated %d\n", get_total_bytes_allocated());

    double percentage_allocated_from_pool = poolSize == 0 ? 0 : (100.0 * allocated_bytes) / poolSize;

    printf("\t\033[34m[Info]\033[0m  Pool (Allocated / Total):          %10d / %lu\n", allocated_bytes, poolSize);
    printf("\t\033[34m[Info]\033[0m  Percentage Allocated:              %10.1lf%%\n", percentage_allocated_from_pool);
    printf("\t\033[34m[Info]\033[0m  Allocated memblock:                %10u\n\n", pool->m_numInitialized);
}

static void pool_init_debug(Pool_t *pool, const size_t poolSize)
{
    printf("\033[0;37m[Call] pool_init_debug\033[0m\n");

    if (!pool->m_memStart)
    {
        perror("Failed to allocate memory for m_memStart");
        pool_destroy(pool);
        exit(0);
    }

    printf("\t\033[0;32m[Debug]\033[0m pool_size given to init pool = %zu\n", poolSize);
    printf("\t\033[0;32m[Debug]\033[0m num blocks = %d\n", pool->m_numOfBlocks);
    printf("\t\033[0;32m[Debug]\033[0m blocks unit size = %d\n", pool->m_sizeOfEachBlock);
    printf("\t\033[0;32m[Debug]\033[0m pool->m_numOfBlocks * pool->m_sizeOfEachBlock = %d\n", pool->m_numOfBlocks * pool->m_sizeOfEachBlock);
    printf("\t\033[0;32m[Debug]\033[0m pool start address : %p\n", (uintptr_t *)pool->m_memStart);
    printf("\t\033[0;32m[Debug]\033[0m pool end address : %p\n", (uintptr_t *)pool->m_memStart + pool->m_numOfBlocks * pool->m_sizeOfEachBlock);
    printf("\t\033[0;32m[Debug]\033[0m start address - end address : %d\n\n", (uint32_t)(pool->m_memStart - (pool->m_memStart + pool->m_numOfBlocks * pool->m_sizeOfEachBlock)));

    printf("\t\033[34m[Info]\033[0m initializing pool...        \n");
    printf("\t\033[34m[Info]\033[0m Pool size                %4u\n", pool->m_numOfBlocks * pool->m_sizeOfEachBlock);
    printf("\t\033[34m[Info]\033[0m Memblock available Qty   %4u\n", pool->m_numOfBlocks);
    printf("\t\033[34m[Info]\033[0m MemBlock unit size       %4u\n", pool->m_sizeOfEachBlock);
    printf("\t\033[34m[Info]\033[0m MemBlock initialized     %4u\n\n", pool->m_numInitialized); 
}

void* memory_alloc_padded (int size, int dtype)
{
    int alignment_size = DEEPC_CPU;
    size_t element_size = sizeof(dtype);
    size_t padded_size = size * element_size;

    if (padded_size % alignment_size != 0) {
        size_t padding = alignment_size - (padded_size % alignment_size);
        padded_size += padding;
    }

    void *allocated_memory = malloc(padded_size);
    if (!allocated_memory) {
        return NULL;
    }
}

void* memory_malloc_aligned(size_t size, size_t alignment) {
    #ifdef _WIN32
        return _aligned_malloc(size, alignment);
    #else
        void* ptr = NULL;
        posix_memalign(&ptr, alignment, size);
        return ptr;
    #endif
}

void memory_free_aligned(void* ptr) {
    #ifdef _WIN32
        _aligned_free(ptr);
    #else
        free(ptr);
    #endif
}

arr_t* arr_alloc(){
    arr_t* data = malloc(sizeof(arr_t));
    if (!data) {
        perror("Error allocating Array structure");
        exit(EXIT_FAILURE);
    }
    return data;
}
