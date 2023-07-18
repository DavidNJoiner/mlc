#include "memory_pool.h"

static GlobalPool global_pool_instance = {0};

void PoolTotalAllocated(Pool *p, size_t* total_allocated, size_t* total_pool_size) {
    // Total pool size
    *total_pool_size = p->blockCount * p->blockSize;
    // Add the size of the blocks array itself
    //*total_pool_size += p->blockCount* sizeof(DEEPC_SIZE_OF_VOID_POINTER);

    // Total memory allocated by all blocks. Subtract memory that has been deallocated.
    // Assuming that each allocation/deallocation is for a single element, not a whole block
    *total_allocated = p->totalObjAllocated * p->elementSize;
    *total_allocated -= p->totalBlockFreed * p->elementSize;
}
/* ------------------------------------------------------------------------------------------------ */
/* PoolReportStats: Print statistics about the pool to the console.
 *
 * p: Pointer to the Pool for which statistics are being reported.
 */
/* ------------------------------------------------------------------------------------------------ */
void PoolReportStats(Pool *p)
{   
    size_t total_allocated, total_pool_size;
    PoolTotalAllocated(p, &total_allocated, &total_pool_size);
    double percentage = 100.0 * total_allocated / total_pool_size;

    printf("----------------------------------------------------\n");
    printf("Pool type:                          %10d\n", p->type);
    printf("Pool (Available / Max):             %10zu / %zu\n", (total_pool_size - total_allocated), total_pool_size);
    printf("Taken from pool:                    %10zu / %.1lf%%\n", total_allocated, percentage);
    printf("Allocated objects:                  %10u\n", p->totalObjAllocated);
    printf("Block freed:                        %10u\n", p->totalBlockFreed);
    printf("New Block allocations:              %10u\n", p->newBlockAllocations);
    printf("Block in-use:                       %10u\n", p->BlocksInUse);
    printf("Blocks Recycled:                    %10u\n", p->totalRecycledBlocks);
    printf("----------------------------------------------------\n\n");
}
/* ------------------------------------------------------------------------------------------------ */
/* GetPool: Retrieve the pool associated with the given ObjectType.
 *
 * type: The ObjectType for which the pool is being retrieved.
 *
 * Returns: Pointer to the corresponding Pool.
 */
/* ------------------------------------------------------------------------------------------------ */
Pool* GetPool(ObjectType type) {
    if (!global_pool_instance.is_initialized) {
        InitializeTensorPool(MAX_OBJ_PER_BLOCK);
    }
    // Return the common pool for any Function subclass
    if (type >= FUNCTION && type <= LAST_FUNCTION_SUBCLASS) {
        return &global_pool_instance.pools[FUNCTION];
    }
    // For non-Function types, return their specific pool
    return &global_pool_instance.pools[type];
}
/* ------------------------------------------------------------------------------------------------ */
/* InitializeTensorPool: Initialize the global memory pool with specific pools for Tensors.
 */
/* ------------------------------------------------------------------------------------------------ */
void InitializeTensorPool(int num_tensors) {
    //if(num_tensors < MAX_OBJ_PER_BLOCK){num_tensors = MAX_OBJ_PER_BLOCK;}
    InitPool(TENSOR, &global_pool_instance.pools[TENSOR], sizeof(Tensor), num_tensors, MAX_OBJ_PER_BLOCK);
    global_pool_instance.is_initialized = true;
}
/* ------------------------------------------------------------------------------------------------ */
/* FreeGlobalTesnorPool: Free the memory of the Tensor pool in the global pool instance and print statistics.
 */
/* ------------------------------------------------------------------------------------------------ */
void FreeTensorPool() {
    FreePool(&global_pool_instance.pools[TENSOR]);
    global_pool_instance.pools[TENSOR] = (Pool){0};
    global_pool_instance.is_initialized = false;
}
/* ---------------------------------------------------------------------------------------------- */
/* InitPool: Initialize a memory pool with the specified characteristics.
 *
 * p: Pointer to the Pool for which memory is being allocated.
 * elementSize: The size of one object.
 * blockSize: The number of object instances that can fit in one block.
 */
/* ---------------------------------------------------------------------------------------------- */
void InitPool(ObjectType type, Pool *p, const uint32_t obj_size, const uint32_t num_obj, const uint32_t max_obj_per_block)
{
    p->type = type;
    p->elementSize = max(obj_size, sizeof(PoolFreed));

    PoolFreeAllBlocks(p);

    p->blockSize = obj_size * MAX_OBJ_PER_BLOCK;                        // Set the individual block size.
    p->blockCount= ceil((float)num_obj / MAX_OBJ_PER_BLOCK);            // Set the block count in the pool.
    p->blocks = malloc(p->blockCount * DEEPC_SIZE_OF_VOID_POINTER);     // Allocate memory from the heap for blocks pointers.

    printf("p->blocks in InitPool =  %p\n", p->blocks);
    
    if (p->blocks == NULL) {
        printf("Error: Failed to allocate memory for blocks\n\n");
        p->blockSize, p->blockCount, p->elementSize = -1;
        exit(1);
    }

    printf("\tInfo : initializing pool    %4d\n", type);
    printf("\tInfo : Pool size            %4d\n", num_obj * obj_size);
    printf("\tInfo : Block size           %4u\n", p->blockSize);
    printf("\tInfo : Objects size         %4d\n", obj_size);
    printf("\tInfo : Pool Capacity(obj)   %4d\n", num_obj); 
    printf("\tInfo : Blocks created       %4u\n\n", p->blockCount);

    for (uint32_t i = 0; i < p->blockCount; ++i)
        p->blocks[i] = NULL;

    p->totalObjAllocated = 0;
    p->totalBlockFreed = 0;
    p->newBlockAllocations = 0;
}
/* ------------------------------------------------------------------------------------------------ */
/* FreePool: Frees all the memory associated with the pool. Should be called when done with the pool.
 *
 * p: Pointer to the Pool to be freed.
 */
/* ------------------------------------------------------------------------------------------------ */
void FreePool(Pool *p)
{
    printf("\nbefore call : FreeTensorPool -> FreePool\n");
    PoolReportStats(p);

    for (uint32_t i = 0; i < p->blockCount; ++i) {
        if (p->blocks[i] == NULL)
            break;
        else
            PoolFreeBlock(p, p->blocks[i]);
    }

    if (p->blocks != NULL) {
        free(p->blocks);
        p->blocks = NULL;
    }

    p->newBlockAllocations = 0;
    p->totalObjAllocated = 0;

    printf("\nafter call : FreeTensorPool -> FreePool\n");
    PoolReportStats(p);
}


#ifndef DISABLE_MEMORY_POOLING



/* ------------------------------------------------------------------------------------------------ */
/* PoolMalloc: Allocate memory for a new object from the specified pool.
 *
 * p: Pointer to the Pool from which memory is being allocated.
 *
 * Returns: Pointer to the allocated MemoryBlock.
 */
/* ------------------------------------------------------------------------------------------------ */
MemoryBlock* PoolMalloc(Pool *p)
{   
    printf("\ncall : PoolMalloc\n");

    if (MAX_OBJ_PER_BLOCK == 0){
        printf("MAX_OBJ_PER_BLOCK must be > than 0.\n");
        exit(1);
    }

    p->totalObjAllocated++; 

    // Checks if there are any previously freed memory blocks that can be reused
    if (p->freed != NULL) {
        MemoryBlock* recycle = (MemoryBlock*)p->freed;
        p->freed = p->freed->nextFree;
        if (p->freed == NULL) {
            p->freedLast = NULL;
        }
        p->totalRecycledBlocks++;
        return recycle;
    }

    printf("blockCount is currently %d\n", p->blockCount);

    // If blockcount is more than 0 all blocks are in use, allocate a new block
    if (p->blockCount != 0 && p->BlocksInUse == p->blockCount) {
        printf("all blocks are full. BlocksInUse = %d / blockCount = %d  Allocating new blocks...\n", p->BlocksInUse, p->blockCount);
        p->BlocksInUse = 0;
        p->block++; 

        // Reallocate memory for p->blocks to accommodate the new block
        p->blocks = realloc(p->blocks, (p->block + 1) * sizeof(*p->blocks));
        if (p->blocks == NULL) {
            printf("Error: Failed to reallocate memory for blocks\n\n");
            FreeTensorPool();
            exit(1);
        }
        p->totalRecycledBlocks ++;
    }
    else{
        MemoryBlock* new_block = NULL;
        // Allocate a new block at address available in the pool (p->blocks pointers array)
        if (p->blocks[p->block] == NULL){
            p->blocks[p->block] = malloc(p->elementSize * p->blockSize);
            printf("Allocated memory for block %d at %p at p->blocks %p \n", p->block, p->blocks[p->block], p->blocks);
            new_block = makeBlock(p, p->blocks[p->block]);
        }else{
            void* new_block_address = p->blocks[p->block] + (p->BlocksInUse * p->elementSize);
            printf("Allocated memory for block %d at %p at p->blocks %p \n", p->block, new_block_address, p->blocks);
            new_block = makeBlock(p, new_block_address);
        }

        p->newBlockAllocations++;
        p->blockCount++;            // Increase the block count
        p->BlocksInUse++;


        //DEBUG-------------------------------------------------------------------------------

        printf("[Debug]\n\tp->blocks[p->block] = %p -> %p\n", p->blocks, p->blocks[p->block]);
        printf("\tp->BlocksInUse = %d\n", p->BlocksInUse);
        printf("\tp->elementSize = %d\n", p->elementSize);
        printf("\tMAX_OBJ_PER_BLOCK = %d\n", MAX_OBJ_PER_BLOCK);
        printf("\tnew_block->size = %d\n", p->elementSize * MAX_OBJ_PER_BLOCK);


        if (p->blocks == NULL) {
            printf("\nError: p->blocks is NULL\n");
            FreeTensorPool();
            exit(1);
        }
        if (sizeof(p->block)/p->blockSize != p->BlocksInUse) {
            printf("\nError: p->block (%d) is out of bounds (p->blockInUse: %d)\n", p->block, p->BlocksInUse);
            FreeTensorPool();
            exit(1);
        }
        if (p->blocks[p->block] == NULL) {
            printf("\nError: p->blocks[p->block] is NULL\n");
            FreeTensorPool();
            exit(1);
        }

        PoolReportStats(p);
        return new_block;
    }
}

MemoryBlock* makeBlock(Pool* p, void* blockAddress){

    printf("making new block at address : %p\n", blockAddress);

    blockAddress = malloc(p->elementSize * p->blockSize);
    if (blockAddress == NULL) {
        printf("Error: Failed to allocate memory for block\n\n");
        FreeTensorPool();
        exit(1);
    }

    MemoryBlock* new_block = (MemoryBlock*)(blockAddress);
    
    if (new_block == NULL) {
        printf("makeBlock : new_block memory allocation failed.\n");
        FreeTensorPool();
        exit(1);
    }
    new_block->size = p->elementSize * MAX_OBJ_PER_BLOCK;
    new_block->ptr = blockAddress;

    return new_block;
}
/* ------------------------------------------------------------------------------------------------ */
/* PoolFreeBlock: Returns a block of memory back to the specified pool so it can be reused.
 *
 * p: Pointer to the Pool from which to free the memory.
 * ptr: Pointer previously returned by PoolMalloc which needs to deallocate that specific memory block.
 */
/* ------------------------------------------------------------------------------------------------ */
void PoolFreeBlock(Pool *p, void *ptr)
{
    // Increment the number of deallocated objects
    p->totalBlockFreed++;

    // If the block's memory has been allocated, free it
    if (ptr != NULL) {
        cudaFree(((MemoryBlock*)ptr)->ptr);
        ((MemoryBlock*)ptr)->ptr = NULL;
    }

    // Prepare the freed block for addition to the freed list
    PoolFreed* freedBlock = (PoolFreed*)ptr;

    // Add the block to the list of freed blocks
    if (p->freed == NULL) {
        p->freed = freedBlock;
        p->freedLast = freedBlock;
    }
    else {
        p->freedLast->nextFree = freedBlock;
        p->freedLast = freedBlock;
    }
    p->freedLast->nextFree = NULL;
}

#endif //DISABLE_MEMORY_POOLING
/* ------------------------------------------------------------------------------------------------ */
/* PoolFreeAllBlocks: Resets the pool, marking all blocks as free.
 * Note: This does not actually free the memory associated with the pool. For that, use FreePool.
 *
 * p: Pointer to the Pool to be reset.
 */
/* ------------------------------------------------------------------------------------------------ */
void PoolFreeAllBlocks(Pool *p)
{
    p->BlocksInUse = 0; 
    p->block = 0; 
    p->freed = NULL;
    p->freedLast = NULL;
}
/* ------------------------------------------------------------------------------------------------ */
/* DeepFreeTensor: Frees all the memory associated with a given Tensor pool, including the      */
/* memory of the Tensors stored in the pool. This function should be used with caution, as it will  */
/* invalidate all Tensors that were allocated from the pool. After calling this function, the pool  */
/* can be reused for new allocations.                                                               */
/*                                                                                                  */
/* p: Pointer to the Tensor Pool to be freed in depth.                                              */
/* ------------------------------------------------------------------------------------------------ */
void DeepFreeTensors(Pool *tensor_pool) {
    // Free the memory of each Tensor in the pool
    for (uint32_t i = 0; i < tensor_pool->blockCount; ++i) {
        if (tensor_pool->blocks[i] != NULL) {
            // Iterate over each Tensor in the block
            for (uint32_t j = 0; j < tensor_pool->blockSize; ++j) {
                // Calculate the address of the Tensor
                Tensor* tensor_ptr = (Tensor*)(tensor_pool->blocks[i] + j * tensor_pool->elementSize);
                // Free the Tensor
                freeTensor(tensor_ptr);
            }
            // Free the block itself
            free(tensor_pool->blocks[i]);
        }
    }

    if (tensor_pool->blocks != NULL) {
        free(tensor_pool->blocks);
        tensor_pool->blocks = NULL;
    }

    FreeAllDatas(); // Free all Datas objects

    // Reset the pool
    tensor_pool->BlocksInUse = tensor_pool->blockSize - 1;
    tensor_pool->block = -1;
    tensor_pool->freed = NULL;
    tensor_pool->freedLast = NULL;
}
/*  ---------------------------------------------------------------*/
/*  freeTensor : Releases the memory allocated for a given tensor. */
/*  ---------------------------------------------------------------*/
void freeTensor(Tensor* t) {
    if (t != NULL) {
        if (t->data != NULL) {
            /* if (t->data->values != NULL) {
                if (t->device->type == CUDA) {
                    cudaFree(t->data->values); 
                } else {
                    printf("%p\n", t->data->values);
                    free(t->data->values);  
                }
                t->data->values = NULL;
            } */
            //free(t->data);
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
        Pool* tensorPool = GetPool(TENSOR);
        PoolFreeBlock(tensorPool, t);
        t = NULL;
    }
}