#include "mempool.h"

void print_memblock_info(MemoryBlock_ptr memblock)
{
    uint32_t i;

    printf("========================================\n");
    printf("MEMBLOCK size: %d\n", BLOCKSIZE);
    printf("MEMBLOCK start @ 0x%08x\n", (unsigned int)(uintptr_t)memblock->m_subblock_array);
    printf("total free: %d\n", total_free());

    for (i = 0; i <= MAX_ORDER; i++)
    {
        print_list_subblock(memblock, i);
    }
}

MemoryBlock_ptr block_alloc(Pool_t *pool)
{
    printf("\n[call] : block_alloc \n");

    // If m_numInitialized is more than 0 or all blocks are in use, allocate memory for new MemoryBlocks
    if (pool->m_numInitialized != 0 && pool->m_numInitialized == pool->m_numOfBlocks)
    {
        printf("\t[Debug] all blocks in use ( num block init %d ). allocating new blocks...\n", pool->m_numInitialized);
        destroy_pool(pool);
        exit(1);
    }

    MemoryBlock_ptr new_block_address;

    if (pool->m_next = pool->m_memStart)
    {
        new_block_address = (MemoryBlock_t *)pool->m_memStart;
    }
    else
    {
        new_block_address = pool->m_next;
    }

    if (new_block_address == NULL)
    {
        printf("\t[Error] Memory allocation for new_block failed\n");
        return NULL;
    }

    size_t freelist = 0;
    // Initialize the freelist array
    for (int i = 0; i <= MAX_ORDER; i++)
    {
        new_block_address->freelist[i] = (SubBlock_ptr)malloc(sizeof(SubBlock_t));
        printf("size freelist = %zu\n", freelist += sizeof(SubBlock_t));
    }

    memset(new_block_address->m_subblock_array, 0, BLOCKSIZE);

    pool->m_numInitialized++;
    pool->m_numFreeBlocks--;
    pool->m_next = (MemoryBlock_t *)((uintptr_t)pool->m_numInitialized * pool->m_sizeOfEachBlock + BLOCKSIZE);

    printf("\t[Info] : Memory Block Allocation Successful !\n");

    total_bytes_allocated += sizeof(new_block_address->freelist);
    add_entry("bloc_alloc", 2, (double)(sizeof(new_block_address->freelist)), 0);
    printf("total_bytes_currently_allocated = %d\n", total_bytes_allocated);

    return new_block_address;
}

void free_block(Pool_t *pool, MemoryBlock_ptr block)
{
    printf("[call] : free_block\n");

    // Check if the block is in the list of allocated blocks
    bool isAllocated = false;
    for (uint32_t i = 0; i <= pool->m_numInitialized; i++)
    {

        if (block == pool->m_memStart + i * BLOCKSIZE)
        {
            isAllocated = true;
            // subblock_free_all(block);
            // free((DEEPC_VOID_POINTER)(block->freelist));

            total_bytes_allocated -= sizeof(block->freelist);
            add_entry("free_bloc", 2, 0.0, (double)(sizeof(block->freelist)));

            pool->m_numFreeBlocks++;
            pool->m_numInitialized--;
        }
    }
}

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