#include "mempool.h"

MemoryBlock_t *memblock_alloc(Pool_t *pool)
{
    // Print a debugging message indicating the function call.
    printf("[Call] memblock_alloc\n");

    // Check if all blocks are in use and handle the situation.
    if (pool->m_numInitialized != 0 && pool->m_numInitialized == pool->m_numOfBlocks)
    {
        printf("[Debug] All blocks are in use (num block init %d). Allocating new blocks...\n", pool->m_numInitialized);
        destroy_pool(pool);
        exit(1);
    }

    // Try to allocate a new MemoryBlock.
    MemoryBlock_t *new_block_address = pool->m_next ? pool->m_next : (MemoryBlock_t *)pool->m_memStart;

    if (!new_block_address)
    {
        printf("[Error] Memory allocation for new_block failed\n");
        return NULL;
    }

    // Update the pool properties.
    pool->m_numInitialized++;
    pool->m_numFreeBlocks--;
    pool->m_next = pool->m_memStart + (pool->m_numInitialized * BLOCKSIZE);

    // Print an informative message about the successful allocation.
    printf("[Info] Memory Block Allocation Successful! Address %p\n", (void *)new_block_address);

    // Update and display memory allocation statistics.
    increase_total_bytes_allocated(BLOCKSIZE);
    add_entry("bloc_alloc", 2, (double)(sizeof(MemoryBlock_t)), 0.0);
    printf("total_bytes_currently_allocated = %d\n", get_total_bytes_allocated());

    return new_block_address;
}
/*
 *   Free a specific block from the pool
 */
void block_free(Pool_t *pool, MemoryBlock_t *block)
{
    // TODO : subblock_free_all(block); free all subblocks in the given block.
    /* for (int i = 0; i <= MAX_ORDER + 1; i++)
    {
        // free(block->freelist[i]);
        block->freelist[i] = NULL;
    } */
    printf("\t\033[34m[Info] Freed block at address %p\033[0m\n", (void *)(block));
    block = NULL;

    printf("\t\033[0;37m[Call] decrease_total_bytes_allocated\033[0m\n");
    decrease_total_bytes_allocated(BLOCKSIZE);
    add_entry("memblock_free", 2, 0.0, (double)(sizeof(MemoryBlock_t)));

    pool->m_numFreeBlocks++;
    pool->m_numInitialized--;
}
/*
 *   Look block match before freeing it from the pool
 */
void memblock_free(Pool_t *pool, MemoryBlock_t *block)
{
    printf("\033[0;37m[Call] memblock_free\033[0m\n");

    bool isAllocated = false;
    for (uint32_t i = 0; i < pool->m_numInitialized; i++)
    {

        printf("\tChecking Initialized bloc %d out of %d indexes\n", i, pool->m_numInitialized - 1);
        printf("\t%p / %p\n", block, pool->m_memStart + i * BLOCKSIZE);

        if (block == pool->m_memStart + i * BLOCKSIZE)
        {
            isAllocated = true;
            block_free(pool, block);
            break; // block was found - break loop
        }
    }
    if (!isAllocated)
    {
        printf("\033[0;31m[Error] The provided block does not belong to the pool!\033[0m\n");
    }
}

/*  -----------------------------------------------------------------------------*/
/*  Data Memory Managment                                                        */
/*  -----------------------------------------------------------------------------*/

void setup_global_data_ptr_array(int initial_capacity)
{
    printf("\033[0;37m[Call] setup_global_data_ptr_array\033[0m\n");
    global_data_ptr_array = (DataPtrArray *)malloc(sizeof(DataPtrArray));
    global_data_ptr_array->data_ptrs = (Data **)malloc(sizeof(Data *) * initial_capacity);
    global_data_ptr_array->count = 0;
    global_data_ptr_array->capacity = initial_capacity;

    data_total_alloc += sizeof(DataPtrArray);
    data_total_alloc += sizeof(sizeof(Data *) * initial_capacity);
}

void add_data_ptr(Data *data_ptr)
{
    printf("\033[0;37m[Call] add_data_ptr\033[0m\n");
    if (global_data_ptr_array->count == global_data_ptr_array->capacity)
    {
        global_data_ptr_array->capacity *= 2;
        global_data_ptr_array->data_ptrs = (Data **)realloc(global_data_ptr_array->data_ptrs, sizeof(Data *) * global_data_ptr_array->capacity);
    }
    global_data_ptr_array->count++;
    global_data_ptr_array->data_ptrs[global_data_ptr_array->count - 1] = data_ptr;
    printf("\t\033[0;32m[Debug]\033[0m Data ptrs count = %d\n", global_data_ptr_array->count);
    printf("\t\033[0;32m[Debug]\033[0m Data pointer added : %p\n", global_data_ptr_array->data_ptrs[global_data_ptr_array->count - 1]);
}
void free_all_data()
{
    printf("\033[0;37m[Call] free_all_data\033[0m\n");
    if (global_data_ptr_array != NULL)
    {
        printf("\t\033[0;32m[Debug]\033[0m global_data_ptr_array is not NULL\n");
        for (uint32_t i = 0; i < global_data_ptr_array->count; i++)
        {
            if (global_data_ptr_array->data_ptrs[i] != NULL)
            {
                printf("\t\033[0;32m[Debug]\033[0m free Datas at address %p\n", global_data_ptr_array->data_ptrs[i]);
                free(global_data_ptr_array->data_ptrs[i]);
                global_data_ptr_array->data_ptrs[i] = NULL;
            }
            data_total_dealloc += sizeof(Data);
        }
        // Free the memory allocated for the array of Data ptrs
        free(global_data_ptr_array->data_ptrs);
        global_data_ptr_array->data_ptrs = NULL;
        // Free the memory allocated for the DataPtrArray object itself
        free(global_data_ptr_array);
        global_data_ptr_array = NULL;

        data_total_dealloc += 2 * sizeof(DEEPC_SIZE_OF_VOID_POINTER);
    }
}