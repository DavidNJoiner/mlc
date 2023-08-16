#include "mempool.h"
#include <math.h>

static GlobalPool_t global_pool_instances = {0};

// MEMBLOCK DEFINITION

pointer_t subblock_AddrFromIndex(const Pool_t *pool, uint32_t i)
{
    return pool->m_memStart + (i * pool->m_sizeOfEachBlock);
}

uint32_t subblock_IndexFromAddr(const Pool_t *pool, const unsigned char *memblock_address)
{
    return (uint32_t)((uintptr_t)memblock_address - (uintptr_t)pool->m_memStart) / pool->m_sizeOfEachBlock;
}

pointer_t subblock_malloc(uint32_t size, MemoryBlock_t *MEMBLOCK)
{
    pointer_t subblock;
    pointer_t memblock;
    uint32_t i, order;

    // Align the size
    size = ALIGN_SIZE(size);

    // calculate minimal order for this size
    i = 0;
    while (BLOCKSIZE * i < size + 1) // one more byte for storing order
        i++;

    order = i = (i < MIN_ORDER) ? MIN_ORDER : i;

    // level up until non-null list found
    for (;; i++)
    {
        if (i > MAX_ORDER)
            return NULL;
        if (MEMBLOCK->freelist[i])
            break;
    }

    // remove the block out of list
    subblock = MEMBLOCK->freelist[i];
    MEMBLOCK->freelist[i] = *(pointer_t *)MEMBLOCK->freelist[i];

    // split until i == order
    while (i-- > order)
    {
        memblock = (unsigned char *)MEMBLOCKOF(subblock, i, MEMBLOCK);
        ((MemoryBlock_t *)memblock)->freelist[i] = memblock;
    }

    // Align the starting address of the block
    subblock = ALIGN_ADDR(subblock);

    // store order in previous byte
    *((uint8_t *)(subblock - 1)) = order;
    return subblock;
}

void subblock_free_all(MemoryBlock_t *MEMBLOCK)
{
    // Loop through the entire memory block
    uintptr_t current_address = (uintptr_t)MEMBLOCK->m_subblock_array;
    uintptr_t end_address = current_address + BLOCKSIZE;

    while (current_address < end_address)
    {
        // Get the current subblock
        pointer_t subblock = (pointer_t)current_address;

        // Fetch order in previous byte
        uint32_t i = *((uint8_t *)(subblock - 1));

        pointer_t memblock;
        pointer_t *p;

        for (;; i++)
        {
            memblock = (unsigned char *)MEMBLOCKOF((SubBlock_t *)subblock, i, MEMBLOCK);
            p = &(((MemoryBlock_t *)memblock)->freelist[i]);

            // Find memblock in list
            while ((*p != NULL) && (*p != memblock))
                p = (pointer_t *)*p;

            // Not found, insert into list
            if (*p != memblock)
            {
                subblock = ((MemoryBlock_t *)memblock)->freelist[i];
                ((MemoryBlock_t *)memblock)->freelist[i] = subblock;
                break;
            }
            // Found, merged block starts from the lower one
            subblock = (subblock < memblock) ? subblock : memblock;
            // Remove buddy out of list
            *p = *(pointer_t *)*p;
        }

        // Move to the next subblock
        current_address += (1 << i);
    }
}

void memblock_init(MemoryBlock_t *memblock)
{
    memset(memblock, 0, sizeof(MemoryBlock_t));
    memblock->freelist[MAX_ORDER] = memblock->m_subblock_array;
}

void memblock_deinit(MemoryBlock_t *memblock)
{
    memblock = NULL;
}

static int count_blocks(uint32_t i)
{

    uint32_t count = 0;
    pointer_t *p = &(((MemoryBlock_t *)(subblock_AddrFromIndex(&global_pool_instances.m_pools[0], i)))->freelist[i]); // Currently only pool 0

    while (*p != NULL)
    {
        count++;
        p = (pointer_t *)*p;
    }
    return count;
}

static int total_free()
{

    int i, bytecount = 0;

    for (i = 0; i <= MAX_ORDER; i++)
    {
        bytecount += count_blocks(i) * BLOCKSIZE * i;
    }
    return bytecount;
}

static void print_list(MemoryBlock_t *MEMBLOCK, uint32_t i)
{

    printf("freelist[%d]: \n", i);

    pointer_t *p = &MEMBLOCK->freelist[i];
    while (*p != NULL)
    {
        printf("    0x%08lx, 0x%08lx\n", (uintptr_t)*p, (uintptr_t)*p - (uintptr_t)MEMBLOCK->m_subblock_array);
        p = (pointer_t *)*p;
    }
}

void print_memblock(MemoryBlock_t *MEMBLOCK)
{

    uint32_t i;

    printf("========================================\n");
    printf("MEMBLOCK size: %d\n", BLOCKSIZE);
    printf("MEMBLOCK start @ 0x%08x\n", (unsigned int)(uintptr_t)MEMBLOCK->m_subblock_array);
    printf("total free: %d\n", total_free());

    for (i = 0; i <= MAX_ORDER; i++)
    {
        print_list(MEMBLOCK, i);
    }
}

// Merge two SubBlocks in a common MemoryBlock
void merge_subblocks(MemoryBlock_t *memblock, SubBlock_t *subblock1, SubBlock_t *subblock2)
{
    // Ensure both SubBlocks are adjacent
    uintptr_t distance = (uintptr_t)subblock2 - (uintptr_t)subblock1;
    if (distance != subblock1->m_SubBlockSize && distance != subblock2->m_SubBlockSize)
    {
        printf("[Error] SubBlocks are not adjacent and cannot be merged.\n");
        return;
    }

    // Merge the SubBlocks
    subblock1->m_SubBlockSize *= 2;
    subblock2->m_SubBlockSize = 0; // Mark the second SubBlock as empty
}

// Remove a SubBlock
void remove_subblock(MemoryBlock_t *memblock, SubBlock_t *subblock)
{
    // Check if its buddy is also free and merge them
    uint32_t subblock_size = (uint32_t)log2((double)subblock->m_SubBlockSize); // No loss, subblocksize are always round.
    uintptr_t buddy_address = (uintptr_t)MEMBLOCKOF(subblock, subblock_size, memblock);
    SubBlock_t *buddy = (SubBlock_t *)buddy_address;
    if (buddy->m_SubBlockSize == subblock->m_SubBlockSize)
    {
        merge_subblocks(memblock, subblock, buddy);
    }
    // Mark the SubBlock as free
    subblock->m_SubBlockSize = 0;
}

// Optimize the layout within the MemoryBlock after deletion
void optimize_layout(MemoryBlock_t *memblock)
{
    // Iterate through the MemoryBlock and try to merge adjacent free SubBlocks
    uintptr_t current_address = (uintptr_t)memblock->m_subblock_array;
    uintptr_t end_address = current_address + BLOCKSIZE;

    while (current_address < end_address)
    {
        SubBlock_t *current_subblock = (SubBlock_t *)current_address;
        if (current_subblock->m_SubBlockSize == 0)
        {
            current_address += sizeof(SubBlock_t); // Move to the next SubBlock
            continue;
        }

        uintptr_t next_address = current_address + current_subblock->m_SubBlockSize;
        SubBlock_t *next_subblock = (SubBlock_t *)next_address;

        // If the next SubBlock is free and of the same size, merge them
        if (next_subblock->m_SubBlockSize == current_subblock->m_SubBlockSize)
        {
            merge_subblocks(memblock, current_subblock, next_subblock);
        }
        else
        {
            current_address = next_address;
        }
    }
}

// END MEMBLOCK DEFINITIONS

void display_pool_stats(Pool_t *pool)
{
    size_t poolAllocated, poolSize;

    poolAllocated = pool->m_numInitialized * pool->m_sizeOfEachBlock;
    poolSize = pool->m_numOfBlocks * pool->m_sizeOfEachBlock;

    double percentage_taken_from_pool = poolAllocated == 0 ? 0 : (100.0 * poolAllocated) / pool->m_numInitialized;

    printf("\t[Info] Pool (Available / Max):             %10zu / %ld\n", (poolSize - poolAllocated), poolSize);
    printf("\t[Info] Taken from pool:                    %10.1lf%%\n", percentage_taken_from_pool);
    printf("\t[Info] Allocated memblock:                 %10u\n", pool->m_numInitialized);
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

void setup_pool(uint8_t pool_instance_index, uint32_t pool_size)
{
    init_pool_(&global_pool_instances.m_pools[pool_instance_index], pool_size);
    global_pool_instances.m_is_initialized = true;
}

static void init_pool_(Pool_t *pool, const uint32_t poolSize)
{
    printf("[call] : init_pool_\n");
    static_assert(alignof(MemoryBlock_t) == 8, "\t[Error] MemoryBlock is not correctly aligned!\n");

    pool->m_numOfBlocks = poolSize / sizeof(MemoryBlock_t);
    pool->m_sizeOfEachBlock = sizeof(MemoryBlock_t);
    pool->m_memStart = (pointer_t)malloc(pool->m_sizeOfEachBlock * pool->m_numOfBlocks);
    pool->m_numFreeBlocks = pool->m_numOfBlocks;
    pool->m_next = pool->m_memStart;
    pool->m_numInitialized = 0;

    printf("\t[Debug] pool blocks start address : %p\n", pool->m_blocks);

    if (pool->m_blocks == NULL)
    {
        printf("\t[Error] Failed to allocate memory for blocks\n");
        pool->m_sizeOfEachBlock, pool->m_numInitialized = 0;
        exit(1);
    }

    printf("\t[Info] : initializing pool    ");
    printf("\t[Info] : Pool size            %4d\n", pool->m_numOfBlocks * pool->m_sizeOfEachBlock);
    printf("\t[Info] : MemBlock size        %4u\n", pool->m_sizeOfEachBlock);
    printf("\t[Info] : Blocks initialized   %4u\n\n", pool->m_numInitialized);
}

void destroy_pool(Pool_t *pool)
{
    printf("[call] : destroy_pool\n");

    free(pool->m_memStart);
    pool->m_memStart = NULL;

    // free all memoryBlock present in the pool
    for (uint32_t i = 0; i < pool->m_numInitialized - 1; ++i)
    {
        if (pool->m_blocks[i] == NULL)
            break;
        else
            free_block(pool, (MemoryBlock_t *)pool->m_blocks[i]);
    }

    // free the pointer list
    if (pool->m_blocks != NULL)
    {
        free(pool->m_blocks);
        pool->m_blocks = NULL;
    }

    display_pool_stats(pool);
}

#ifndef DISABLE_MEMORY_POOLING

MemoryBlock_t *block_alloc(Pool_t *pool)
{
    printf("\n[call] : block_alloc \n");

    // Checks if there are any previously freed memory blocks that can be reused
    if (pool->m_freed != NULL)
    {
        MemoryBlock_t *recycle = (MemoryBlock_t *)pool->m_freed;
        pool->m_freed = pool->m_freed->m_nextFree;
        if (pool->m_freed == NULL)
        {
            pool->m_freedLast = NULL;
        }
        return recycle;
    }

    // If m_numInitialized is more than 0 or all blocks are in use, allocate memory for new MemoryBlocks
    if (pool->m_numInitialized != 0 && pool->m_numInitialized == pool->m_numOfBlocks)
    {
        printf("\t[Debug] all blocks in use ( blocks_in_use = %d ). allocating new blocks...\n", pool->m_numInitialized);
    }

    MemoryBlock_t *new_block = NULL;

    // Allocate a new block at the next available address in the pool
    unsigned char *new_block_address = pool->m_memStart + (pool->m_numInitialized * pool->m_sizeOfEachBlock);
    printf("\t[Debug] Allocated memory for block %p \n", new_block_address);

    // Update the pool's metadata
    pool->m_blocks[pool->m_numInitialized] = new_block_address;
    pool->m_numInitialized++;

    return (MemoryBlock_t *)new_block_address;
}

MemoryBlock_t *create_block(Pool_t *pool)
{
    printf("\n[call] : create_block\n");

    MemoryBlock_t *new_block = block_alloc(pool);
    if (new_block == NULL)
    {
        printf("\t[Error] Failed to create a new block\n");
    }

    // Update pool's metadata
    pool->m_numOfBlocks++;
    pool->m_numFreeBlocks--;
    return new_block;
}

void free_block(Pool_t *pool, MemoryBlock_t *block)
{
    printf("[call] : free_block\n");

    // Check if the block is in the list of allocated blocks
    bool isAllocated = false;
    for (uint32_t i = 0; i < pool->m_numInitialized; i++)
    {
        if (pool->m_blocks[i] == (uint8_t *)block)
        {
            isAllocated = true;
            break;
        }
    }

    if (isAllocated)
    {
        subblock_free_all(block);

        // Update pool's metadata
        pool->m_numFreeBlocks++;
        pool->m_numInitialized--;
    }
}

#endif // DISABLE_MEMORY_POOLING

/*  -----------------------------------------------------------------------------*/
/*  Data Memory Managment                                                        */
/*  -----------------------------------------------------------------------------*/
void setup_global_data_ptr_array(int initial_capacity)
{
    printf("\n[call] : setup_global_data_ptr_array\n");
    global_data_ptr_array = (DataPtrArray *)malloc(sizeof(DataPtrArray));
    global_data_ptr_array->data_ptrs = (Data **)malloc(sizeof(Data *) * initial_capacity);
    global_data_ptr_array->count = 0;
    global_data_ptr_array->capacity = initial_capacity;

    total_data_allocated += sizeof(DataPtrArray);
    total_data_allocated += sizeof(sizeof(Data *) * initial_capacity);
}

void add_data_ptr(Data *data_ptr)
{
    printf("\n[call] : add_data_ptr :\n");
    if (global_data_ptr_array->count == global_data_ptr_array->capacity)
    {
        global_data_ptr_array->capacity *= 2;
        global_data_ptr_array->data_ptrs = (Data **)realloc(global_data_ptr_array->data_ptrs, sizeof(Data *) * global_data_ptr_array->capacity);
    }
    global_data_ptr_array->count++;
    global_data_ptr_array->data_ptrs[global_data_ptr_array->count - 1] = data_ptr;
    printf("\t[DEBUG] Data ptrs count = %d\n", global_data_ptr_array->count);
    printf("\t[DEBUG] Data pointer added : %p\n", global_data_ptr_array->data_ptrs[global_data_ptr_array->count - 1]);
}
void free_all_data()
{
    printf("\n[call] : free_all_data\n");
    if (global_data_ptr_array != NULL)
    {
        printf("\t[DEBUG] global_data_ptr_array is not NULL\n");
        for (uint32_t i = 0; i < global_data_ptr_array->count; i++)
        {
            if (global_data_ptr_array->data_ptrs[i] != NULL)
            {
                printf("\t[DEBUG] free Datas at address %p\n", global_data_ptr_array->data_ptrs[i]);
                free(global_data_ptr_array->data_ptrs[i]);
                global_data_ptr_array->data_ptrs[i] = NULL;
            }
            total_data_deallocated += sizeof(Data);
        }
        // Free the memory allocated for the array of Data ptrs
        free(global_data_ptr_array->data_ptrs);
        global_data_ptr_array->data_ptrs = NULL;
        // Free the memory allocated for the DataPtrArray object itself
        free(global_data_ptr_array);
        global_data_ptr_array = NULL;

        total_data_deallocated += 2 * sizeof(DEEPC_SIZE_OF_VOID_POINTER);
    }
}