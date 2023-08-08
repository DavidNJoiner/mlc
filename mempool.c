#include "mempool.h"

static GlobalPool global_pool_instance = {0};
int total_allocated = 0;
int total_deallocated = 0;

/* ------------------------------------------------------------------------------------------------ */
/* display_pool_stats: Print statistics about the pool to the console.
 *
 * p: Pointer to the Pool for which statistics are being reported.
 */
/* ------------------------------------------------------------------------------------------------ */
void display_pool_stats(Pool *p)
{   
    size_t total_allocated_block_size, total_pool_filled;
    size_t poolmaxsize = MAX_OBJ_PER_BLOCK * p->element_size;

    total_pool_filled = p->block_count * p->block_size;
    total_allocated_block_size = p->blocks_in_use * p->element_size;

    double percentage = total_pool_filled == 0 ? 0 : (100.0 * total_allocated_block_size) / total_pool_filled;

    printf("\n\t[Info] Pool type:                          %10d\n", p->type);
    printf("\t[Info] Pool (Available / Max):             %10zu / %ld\n", (poolmaxsize - total_allocated_block_size), poolmaxsize);
    printf("\t[Info] Taken from pool:                    %.1lf%%\n", percentage);
    printf("\t[Info] Allocated objects:                  %10u\n", p->total_obj_allocated);
    printf("\t[Info] Block freed:                        %10u\n", p->total_block_freed);
    printf("\t[Info] New Block allocations:              %10u\n", p->new_block_allocations);
    printf("\t[Info] Block in-use:                       %10u\n", p->blocks_in_use);
    printf("\t[Info] Blocks Recycled:                    %10u\n\n", p->total_recycled_blocks);

}

/* ------------------------------------------------------------------------------------------------ */
/* fetch_pool: Retrieve the pool associated with the given ObjectType.
 *
 * type: The ObjectType for which the pool is being retrieved.
 *
 * Returns: Pointer to the corresponding Pool.
 */
/* ------------------------------------------------------------------------------------------------ */
Pool* fetch_pool(ObjectType type) {
    if (!global_pool_instance.is_initialized) {
        setup_tensor_pool(MAX_OBJ_PER_BLOCK);
    }
    if (type >= FUNCTION && type <= LAST_FUNCTION_SUBCLASS) {
        return &global_pool_instance.pools[FUNCTION];
    }
    return &global_pool_instance.pools[TENSOR];
}
/* ------------------------------------------------------------------------------------------------ */
/* setup_tensor_pool: Initialize the global memory pool with specific pools for Tensors.
 */
/* ------------------------------------------------------------------------------------------------ */
void setup_tensor_pool(int num_tensors) {
    initialize_pool(TENSOR, &global_pool_instance.pools[TENSOR], sizeof(Tensor), num_tensors, MAX_OBJ_PER_BLOCK);
    global_pool_instance.is_initialized = true;
}
/* ------------------------------------------------------------------------------------------------ */
/* destroy_tensor_pool: Free the memory of the Tensor pool in the global pool instance and print statistics.
 */
/* ------------------------------------------------------------------------------------------------ */
void destroy_tensor_pool() {
    destroy_pool(&global_pool_instance.pools[TENSOR]);
    global_pool_instance.pools[TENSOR] = (Pool){0};
    global_pool_instance.is_initialized = false;
}
/* ---------------------------------------------------------------------------------------------- */
/* initialize_pool: Initialize a memory pool with the specified characteristics.
 *
 * p: Pointer to the Pool for which memory is being allocated.
 * element_size: The size of one object.
 * block_size: The number of object instances that can fit in one block.
 */
/* ---------------------------------------------------------------------------------------------- */
void initialize_pool(ObjectType type, Pool *p, const uint32_t obj_size, const uint32_t num_obj, const uint32_t obj_per_block)
{
    p->type = type;
    p->element_size = max(obj_size, sizeof(PoolFreed));

    free_all_blocks(p);

    p->block_size = obj_size * obj_per_block;                            // Set the individual block size.
    p->block_count= ceil((float)num_obj / obj_per_block);                // Set the block count in the pool.
    p->blocks = malloc(p->block_count * DEEPC_SIZE_OF_VOID_POINTER);     // Allocate memory from the heap for blocks pointers.

    total_allocated += p->block_count * DEEPC_SIZE_OF_VOID_POINTER;

    printf("p->blocks in InitPool = %p\n", p->blocks);
    
    if (p->blocks == NULL) {
        printf("Error: Failed to allocate memory for blocks\n\n");
        p->block_size, p->block_count, p->element_size = 0;
        exit(1);
    }

    printf("\t[Info] : initializing pool    %4d\n", type);
    printf("\t[Info] : Pool size            %4d\n", p->block_count * p->block_size);
    printf("\t[Info] : Block size           %4u\n", p->block_size);
    printf("\t[Info] : Objects size         %4d\n", obj_size);
    printf("\t[Info] : Blocks created       %4u\n\n", p->block_count);

    for (uint32_t i = 0; i < p->block_count; ++i)
        p->blocks[i] = NULL;

    p->total_obj_allocated = 0;
    p->total_block_freed = 0;
    p->new_block_allocations = 0;
}
/* ------------------------------------------------------------------------------------------------ */
/* destroy_pool: Frees all the memory associated with the pool. Should be called when done with the pool.
 *
 * p: Pointer to the Pool to be freed.
 */
/* ------------------------------------------------------------------------------------------------ */
void destroy_pool(Pool *p)
{
    // free all memoryBlock present in the pool
    for (uint32_t i = 0; i < p->block_count; ++i) {
        if (p->blocks[i] == NULL)
            break;
        else
            free_block(p, p->blocks[i]);
    }

    // free the pointer list
    if (p->blocks != NULL) {
        free(p->blocks);
        p->blocks = NULL;
        total_deallocated += DEEPC_SIZE_OF_VOID_POINTER;
    }

    free_all_blocks(p);

    printf("\nafter call : destroy_tensor_pool -> destroy_pool\n");
    display_pool_stats(p);

    printf("total_blocks_allocated %d bytes\n", total_allocated);
    printf("total_blocks_deallocated %d bytes\n", total_deallocated);
    printf("total_data_allocated %d bytes\n", total_data_allocated);
    printf("total_data_deallocated %d bytes\n", total_data_deallocated);
}


#ifndef DISABLE_MEMORY_POOLING

/* ------------------------------------------------------------------------------------------------ */
/* allocate_block: Allocate memory for a new object from the specified pool.
 *
 * p: Pointer to the Pool from which memory is being allocated.
 *
 * Returns: Pointer to the allocated MemoryBlock.
 */
/* ------------------------------------------------------------------------------------------------ */
MemoryBlock* allocate_block(Pool *p)
{   
    printf("\ncall allocate_block :\n");

    if (MAX_OBJ_PER_BLOCK == 0){
        printf("MAX_OBJ_PER_BLOCK must be > than 0.\n");
        exit(1);
    }

    p->total_obj_allocated++; 

    // Checks if there are any previously freed memory blocks that can be reused
    if (p->freed != NULL) {
        MemoryBlock* recycle = (MemoryBlock*)p->freed;
        p->freed = p->freed->next_free;
        if (p->freed == NULL) {
            p->freed_last = NULL;
        }
        p->total_recycled_blocks++;
        return recycle;
    }

    printf("\t[Debug] BlockCount is currently %d\n", p->block_count);

    // If blockcount is more than 0 all blocks are in use, allocate a new block
    if (p->block_count != 0 && p->blocks_in_use == p->block_count) {
        printf("all blocks are full. blocks_in_use = %d / blockCount = %d  Allocating new blocks...\n", p->blocks_in_use, p->block_count);
        p->blocks_in_use = 0;
        p->block++; 

        // Reallocate memory for p->blocks to accommodate the new block
        p->blocks = realloc(p->blocks, (p->block + 1) * sizeof(*p->blocks));
        if (p->blocks == NULL) {
            printf("Error: Failed to reallocate memory for blocks\n\n");
            destroy_tensor_pool();
            exit(1);
        }
        p->total_recycled_blocks ++;
    }
    else{
        MemoryBlock* new_block = NULL;
        // Allocate a new block at address available in the pool (p->blocks pointers array)
        // If the first block is uninitialized
        if (p->blocks[p->block] == NULL){
            p->blocks[p->block] = malloc(DEEPC_SIZE_OF_VOID_POINTER);
            printf("\t[Debug] First block : Allocated memory for block %p \n", p->blocks[p->block]);
            new_block = create_block(p, p->blocks[p->block]);
            total_allocated += (DEEPC_SIZE_OF_VOID_POINTER);
        }else{
            void* new_block_address = p->blocks[p->block] + (p->blocks_in_use * p->element_size);
            printf("\t[Debug] Adding to existing block : Allocated memory for block %p \n", new_block_address);
            new_block = create_block(p, new_block_address);
            p->block_count++;
        }

        p->new_block_allocations++;
        p->blocks_in_use++;

        //DEBUG-------------------------------------------------------------------------------

        printf("\t[Debug] New MemoryBlock allocated at p->blocks[p->block] = %p \n", p->blocks[p->block]);
   
        if (p->blocks == NULL) {
            printf("\nError: p->blocks is NULL\n");
            destroy_tensor_pool();
            exit(1);
        }
        /* if (p->total_obj_allocated/p->block_size != p->blocks_in_use) {
            printf("\nError: p->block (%d) is out of bounds (p->blockInUse: %d)\n", p->block, p->blocks_in_use);
            destroy_tensor_pool();
            exit(1);
        } */
        if (p->blocks[p->block] == NULL) {
            printf("\nError: p->blocks[p->block] is NULL\n");
            destroy_tensor_pool();
            exit(1);
        }
        display_pool_stats(p);

        return new_block;
    }
}

MemoryBlock* create_block(Pool* p, void* blockAddress){
    printf("\ncall create_block :\n");
    printf("\t[Debug] MemoryBlock at address : %p\n", blockAddress);
    if (blockAddress == NULL) {
        printf("Error: Failed to allocate memory for MemoryBlock at address NULL\n");
        destroy_tensor_pool();
        exit(1);
    }

    MemoryBlock* new_block = (MemoryBlock*)(blockAddress);
    
    if (new_block == NULL) {
        printf("\t[Debug] create_block : new MemoryBlock allocation at address %p failed.\n", blockAddress);
        destroy_tensor_pool();
        exit(1);
    }
    new_block->size = p->block_size;
    new_block->ptr = blockAddress;

    total_allocated += new_block->size;

    return new_block;
}
/* ------------------------------------------------------------------------------------------------ */
/* free_block: Returns a block of memory back to the specified pool so it can be reused.
 *
 * p: Pointer to the Pool from which to free the memory.
 * ptr: Pointer previously returned by PoolMalloc which needs to deallocate that specific memory block.
 */
/* ------------------------------------------------------------------------------------------------ */
void free_block(Pool *p, void *ptr)
{
    // If the block's memory has been allocated, free it
    if (ptr != NULL ) {
        total_deallocated += ((MemoryBlock*)ptr)->size;
        total_deallocated += sizeof(DEEPC_SIZE_OF_VOID_POINTER);
        free(((MemoryBlock*)ptr)->ptr);
        ((MemoryBlock*)ptr)->ptr = NULL;
    }

    // Prepare the freed block for addition to the freed list
    PoolFreed* freedBlock = (PoolFreed*)ptr;

    // Add the block to the list of freed blocks
    if (p->freed == NULL) {
        p->freed = freedBlock;
        p->freed_last = freedBlock;
    }
    else {
        p->freed_last->next_free = freedBlock;
        p->freed_last = freedBlock;
    }
    p->freed_last->next_free = NULL;

    p->blocks_in_use--;
    p->block_count--;
    p->total_block_freed ++;
}


#endif //DISABLE_MEMORY_POOLING

/* ------------------------------------------------------------------------------------------------ */
/* PoolFreeAllBlocks: Resets the pool, marking all blocks as free.
 * Note: This does not actually free the memory associated with the pool. For that, use FreePool.
 *
 * p: Pointer to the Pool to be reset.
 */
/* ------------------------------------------------------------------------------------------------ */
void free_all_blocks(Pool *p)
{
    p->blocks_in_use = 0; 
    p->block = 0; 
    p->freed = NULL;
    p->freed_last = NULL;
}
/* ------------------------------------------------------------------------------------------------ */
/* free_all_tensors: Frees all the memory associated with a given Tensor pool, including the      */
/* memory of the Tensors stored in the pool. This function should be used with caution, as it will  */
/* invalidate all Tensors that were allocated from the pool. After calling this function, the pool  */
/* can be reused for new allocations.                                                               */
/*                                                                                                  */
/* p: Pointer to the Tensor Pool to be freed in depth.                                              */
/* ------------------------------------------------------------------------------------------------ */
void free_all_tensors() {
    printf("\ncall free_all_tensors :\n");
    printf("\t[DEBUG] freeing Tensors...\n");

    Pool* tensor_pool = fetch_pool(TENSOR);
    printf("\nbefore call : free_all_tensors\n");
    display_pool_stats(tensor_pool);

    // Free the memory of each Tensor in the pool
    for (uint32_t i = 0; i < tensor_pool->block_count; ++i) {
        // Iterate over each block in the pool
        if (tensor_pool->blocks[i] != NULL) {
            // Iterate over each Tensor in the block (MAX_OBJ_PER_BLOCK gives us the number of Tensors in a block)
            for (uint32_t j = 0; j < MAX_OBJ_PER_BLOCK-1; ++j) {
                // Calculate the address of the Tensor
                if (tensor_pool->blocks[i] + j * tensor_pool->element_size != NULL){
                    Tensor* tensor_ptr = (Tensor*)(tensor_pool->blocks[i] + (j * tensor_pool->element_size));
                    free_tensor(tensor_ptr);
                }
            }
            // Free the block itself
            free(tensor_pool->blocks[i]);
        }
    }

    if (tensor_pool->blocks != NULL) {
        free(tensor_pool->blocks);
        total_deallocated += sizeof(tensor_pool->blocks);
        tensor_pool->blocks = NULL;
    }

    printf("\nin call : free_all_tensors\n");
    display_pool_stats(tensor_pool);

    free_all_data(); // Free all Datas objects

    // Reset the pool
    tensor_pool->blocks_in_use = tensor_pool->block_size - 1;
    tensor_pool->block = -1;
    tensor_pool->freed = NULL;
    tensor_pool->freed_last = NULL;
}
/*  ---------------------------------------------------------------*/
/*  free_tensor : Releases the memory allocated for a given tensor. */
/*  ---------------------------------------------------------------*/
void free_tensor(Tensor* t) {
    printf("\ncall free_tensor :\n");
    printf("\t[DEBUG] freeing a Tensor %p\n", (void*)t);
    if (t != NULL) {
        if (t->data != NULL) {
            t->data = NULL; // freeing only the pointer to the Data. the data is cleaned later because some other Tensors might use it.
        }
        if (t->gradient != NULL) {
            if (t->device->type == CUDA) {
                cudaFree(t->gradient); 
            } else {
                free(t->gradient); 
            }
            t->gradient = NULL;
        }
        Pool* tensorPool = fetch_pool(TENSOR);
        free_block(tensorPool, t);
        t = NULL;
    }
}

/*  -----------------------------------------------------------------------------*/
/*  Data Memory Managment                                                        */
/*  -----------------------------------------------------------------------------*/
void setup_global_data_ptr_array(int initial_capacity) {
    printf("\ncall setup_global_data_ptr_array :\n");
    global_data_ptr_array = (DataPtrArray*)malloc(sizeof(DataPtrArray));
    global_data_ptr_array->data_ptrs = (Data**)malloc(sizeof(Data*) * initial_capacity);
    global_data_ptr_array->count = 0;
    global_data_ptr_array->capacity = initial_capacity;

    total_data_allocated += sizeof(DataPtrArray);
    total_data_allocated += sizeof(sizeof(Data*) * initial_capacity);
}

void add_data_ptr(Data* data_ptr) {
    printf("\ncall add_data_ptr :\n");
    if (global_data_ptr_array->count == global_data_ptr_array->capacity) {
        global_data_ptr_array->capacity *= 2;
        global_data_ptr_array->data_ptrs = (Data**)realloc(global_data_ptr_array->data_ptrs, sizeof(Data*) * global_data_ptr_array->capacity);
    }
    global_data_ptr_array->count++;
    global_data_ptr_array->data_ptrs[global_data_ptr_array->count-1] = data_ptr;
    printf("\t[DEBUG] Data ptrs count = %d\n", global_data_ptr_array->count);
    printf("\t[DEBUG] Data ptr added : %p\n", global_data_ptr_array->data_ptrs[global_data_ptr_array->count-1]);
}
void free_all_data() {
    printf("\ncall free_all_data : \n");
    if (global_data_ptr_array != NULL) {
        printf("\t[DEBUG] global_data_ptr_array is not NULL\n");
        for (int i = 0; i < global_data_ptr_array->count; i++) {
            if (global_data_ptr_array->data_ptrs[i] != NULL) {
                printf("\t[DEBUG] free Datas at address %p\n", global_data_ptr_array->data_ptrs[i]);
                free(global_data_ptr_array->data_ptrs[i]);
                global_data_ptr_array->data_ptrs[i] = NULL;
            }
            total_data_deallocated += sizeof(Data);
        }
        // Free the memory allocated for the array of Data pointers
        free(global_data_ptr_array->data_ptrs);
        global_data_ptr_array->data_ptrs = NULL;
        // Free the memory allocated for the DataPtrArray object itself
        free(global_data_ptr_array);
        global_data_ptr_array = NULL;

        total_data_deallocated += 2*sizeof(DEEPC_SIZE_OF_VOID_POINTER);
    }
}