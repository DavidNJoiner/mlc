#include "mempool.h"

static GlobalPool global_pool_instance = {0};
int total_allocated = 0;
int total_deallocated = 0;

void calculatePoolStats(Pool *p, size_t* total_allocated, size_t* total_pool_size) {
    *total_pool_size = p->blockCount * p->blockSize;
    *total_allocated = p->BlocksInUse * p->elementSize;
}
/* ------------------------------------------------------------------------------------------------ */
/* displayPoolStats: Print statistics about the pool to the console.
 *
 * p: Pointer to the Pool for which statistics are being reported.
 */
/* ------------------------------------------------------------------------------------------------ */
void displayPoolStats(Pool *p)
{   
    size_t num_allocated_block, total_pool_size;
    calculatePoolStats(p, &num_allocated_block, &total_pool_size);

    double percentage = total_pool_size == 0 ? 0 : (100.0 * num_allocated_block) / total_pool_size;

    printf("----------------------------------------------------\n");
    printf("Pool type:                          %10d\n", p->type);
    printf("Pool (Available / Max):             %10zu / %ld\n", total_pool_size == 0 ? 0 : (total_pool_size - num_allocated_block), total_pool_size);
    printf("Taken from pool:                    %10ld / %.1lf%%\n", num_allocated_block, percentage);
    printf("Allocated objects:                  %10u\n", p->totalObjAllocated);
    printf("Block freed:                        %10u\n", p->totalBlockFreed);
    printf("New Block allocations:              %10u\n", p->newBlockAllocations);
    printf("Block in-use:                       %10u\n", p->BlocksInUse);
    printf("Blocks Recycled:                    %10u\n", p->totalRecycledBlocks);
    printf("----------------------------------------------------\n\n");
}

/* ------------------------------------------------------------------------------------------------ */
/* fetchPool: Retrieve the pool associated with the given ObjectType.
 *
 * type: The ObjectType for which the pool is being retrieved.
 *
 * Returns: Pointer to the corresponding Pool.
 */
/* ------------------------------------------------------------------------------------------------ */
Pool* fetchPool(ObjectType type) {
    if (!global_pool_instance.is_initialized) {
        setupTensorPool(MAX_OBJ_PER_BLOCK);
    }

    if (type >= FUNCTION && type <= LAST_FUNCTION_SUBCLASS) {
        return &global_pool_instance.pools[FUNCTION];
    }

    return &global_pool_instance.pools[type];
}
/* ------------------------------------------------------------------------------------------------ */
/* setupTensorPool: Initialize the global memory pool with specific pools for Tensors.
 */
/* ------------------------------------------------------------------------------------------------ */
void setupTensorPool(int num_tensors) {

    initializePool(TENSOR, &global_pool_instance.pools[TENSOR], sizeof(Tensor), num_tensors, MAX_OBJ_PER_BLOCK);
    global_pool_instance.is_initialized = true;
}
/* ------------------------------------------------------------------------------------------------ */
/* destroyTensorPool: Free the memory of the Tensor pool in the global pool instance and print statistics.
 */
/* ------------------------------------------------------------------------------------------------ */
void destroyTensorPool() {
    destroyPool(&global_pool_instance.pools[TENSOR]);
    global_pool_instance.pools[TENSOR] = (Pool){0};
    global_pool_instance.is_initialized = false;
}
/* ---------------------------------------------------------------------------------------------- */
/* initializePool: Initialize a memory pool with the specified characteristics.
 *
 * p: Pointer to the Pool for which memory is being allocated.
 * elementSize: The size of one object.
 * blockSize: The number of object instances that can fit in one block.
 */
/* ---------------------------------------------------------------------------------------------- */
void initializePool(ObjectType type, Pool *p, const uint32_t obj_size, const uint32_t num_obj, const uint32_t obj_per_block)
{
    p->type = type;
    p->elementSize = max(obj_size, sizeof(PoolFreed));

    freeAllBlocks(p);

    p->blockSize = obj_size * obj_per_block;                        // Set the individual block size.
    p->blockCount= ceil((float)num_obj / obj_per_block);            // Set the block count in the pool.
    p->blocks = malloc(p->blockCount * DEEPC_SIZE_OF_VOID_POINTER);     // Allocate memory from the heap for blocks pointers.

    total_allocated += p->blockCount * DEEPC_SIZE_OF_VOID_POINTER;

    printf("p->blocks in InitPool = %p\n", p->blocks);
    
    if (p->blocks == NULL) {
        printf("Error: Failed to allocate memory for blocks\n\n");
        p->blockSize, p->blockCount, p->elementSize = 0;
        exit(1);
    }

    printf("\tInfo : initializing pool    %4d\n", type);
    printf("\tInfo : Pool size            %4d\n", p->blockCount * p->blockSize);
    printf("\tInfo : Block size           %4u\n", p->blockSize);
    printf("\tInfo : Objects size         %4d\n", obj_size);
    printf("\tInfo : Blocks created       %4u\n\n", p->blockCount);

    for (uint32_t i = 0; i < p->blockCount; ++i)
        p->blocks[i] = NULL;

    p->totalObjAllocated = 0;
    p->totalBlockFreed = 0;
    p->newBlockAllocations = 0;
}
/* ------------------------------------------------------------------------------------------------ */
/* destroyPool: Frees all the memory associated with the pool. Should be called when done with the pool.
 *
 * p: Pointer to the Pool to be freed.
 */
/* ------------------------------------------------------------------------------------------------ */
void destroyPool(Pool *p)
{
    printf("\nbefore call : destroyTensorPool -> destroyPool\n");
    displayPoolStats(p);

    // free all memoryBlock present in the pool
    for (uint32_t i = 0; i < p->blockCount; ++i) {
        if (p->blocks[i] == NULL)
            break;
        else
            freeBlock(p, p->blocks[i]);
    }

    // free the pointer list
    if (p->blocks != NULL) {
        free(p->blocks);
        p->blocks = NULL;
        total_deallocated += DEEPC_SIZE_OF_VOID_POINTER;
    }

    p->blockCount = 0;

    printf("\nafter call : destroyTensorPool -> destroyPool\n");
    displayPoolStats(p);

    printf("total_allocated %d bytes\n", total_allocated);
    printf("total_deallocated %d bytes\n", total_deallocated);
}


#ifndef DISABLE_MEMORY_POOLING

/* ------------------------------------------------------------------------------------------------ */
/* allocateBlock: Allocate memory for a new object from the specified pool.
 *
 * p: Pointer to the Pool from which memory is being allocated.
 *
 * Returns: Pointer to the allocated MemoryBlock.
 */
/* ------------------------------------------------------------------------------------------------ */
MemoryBlock* allocateBlock(Pool *p)
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

    printf("| blockCount is currently %d\n", p->blockCount);

    // If blockcount is more than 0 all blocks are in use, allocate a new block
    if (p->blockCount != 0 && p->BlocksInUse == p->blockCount) {
        printf("all blocks are full. BlocksInUse = %d / blockCount = %d  Allocating new blocks...\n", p->BlocksInUse, p->blockCount);
        p->BlocksInUse = 0;
        p->block++; 

        // Reallocate memory for p->blocks to accommodate the new block
        p->blocks = realloc(p->blocks, (p->block + 1) * sizeof(*p->blocks));
        if (p->blocks == NULL) {
            printf("Error: Failed to reallocate memory for blocks\n\n");
            destroyTensorPool();
            exit(1);
        }
        p->totalRecycledBlocks ++;
    }
    else{
        MemoryBlock* new_block = NULL;
        // Allocate a new block at address available in the pool (p->blocks pointers array)
        // If the first block is uninitialized
        if (p->blocks[p->block] == NULL){
            p->blocks[p->block] = malloc(DEEPC_SIZE_OF_VOID_POINTER);
            total_allocated += (DEEPC_SIZE_OF_VOID_POINTER);
            printf("| allocated memory for block %d at %p at p->blocks %p \n", p->block, p->blocks[p->block], p->blocks);
            new_block = createBlock(p, p->blocks[p->block]);
        }else{
            void* new_block_address = p->blocks[p->block] + (p->BlocksInUse * p->elementSize);
            printf("| allocated memory for block %d at %p at p->blocks %p \n", p->block, new_block_address, p->blocks);
            new_block = createBlock(p, new_block_address);
            p->blockCount++;
        }

        p->newBlockAllocations++;
        p->BlocksInUse++;

        //DEBUG-------------------------------------------------------------------------------

        printf("[Debug]\n\tp->blocks[p->block] = %p -> %p\n", p->blocks, p->blocks[p->block]);
        printf("\tp->BlocksInUse = %d\n", p->BlocksInUse);
        printf("\tp->elementSize = %d\n", p->elementSize);
        printf("\tMAX_OBJ_PER_BLOCK = %d\n", MAX_OBJ_PER_BLOCK);
        printf("\tnew_block->size = %d\n", p->elementSize * MAX_OBJ_PER_BLOCK);

        if (p->blocks == NULL) {
            printf("\nError: p->blocks is NULL\n");
            destroyTensorPool();
            exit(1);
        }
        if (p->totalObjAllocated/p->blockSize != p->BlocksInUse) {
            printf("\nError: p->block (%d) is out of bounds (p->blockInUse: %d)\n", p->block, p->BlocksInUse);
            destroyTensorPool();
            exit(1);
        }
        if (p->blocks[p->block] == NULL) {
            printf("\nError: p->blocks[p->block] is NULL\n");
            destroyTensorPool();
            exit(1);
        }
        displayPoolStats(p);
        return new_block;
    }
}

MemoryBlock* createBlock(Pool* p, void* blockAddress){

    printf("| making new block at address : %p\n\n", blockAddress);
    if (blockAddress == NULL) {
        printf("Error: Failed to allocate memory for block\n\n");
        destroyTensorPool();
        exit(1);
    }

    MemoryBlock* new_block = (MemoryBlock*)(blockAddress);
    
    if (new_block == NULL) {
        printf("makeBlock : new_block memory allocation failed.\n");
        destroyTensorPool();
        exit(1);
    }
    new_block->size = p->blockSize;
    new_block->ptr = blockAddress;

    total_allocated += new_block->size;

    return new_block;
}
/* ------------------------------------------------------------------------------------------------ */
/* freeBlock: Returns a block of memory back to the specified pool so it can be reused.
 *
 * p: Pointer to the Pool from which to free the memory.
 * ptr: Pointer previously returned by PoolMalloc which needs to deallocate that specific memory block.
 */
/* ------------------------------------------------------------------------------------------------ */
void freeBlock(Pool *p, void *ptr)
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
        p->freedLast = freedBlock;
    }
    else {
        p->freedLast->nextFree = freedBlock;
        p->freedLast = freedBlock;
    }
    p->freedLast->nextFree = NULL;

    p->BlocksInUse--;
    p->blockCount--;
    p->totalBlockFreed ++;
}


#endif //DISABLE_MEMORY_POOLING

/* ------------------------------------------------------------------------------------------------ */
/* PoolFreeAllBlocks: Resets the pool, marking all blocks as free.
 * Note: This does not actually free the memory associated with the pool. For that, use FreePool.
 *
 * p: Pointer to the Pool to be reset.
 */
/* ------------------------------------------------------------------------------------------------ */
void freeAllBlocks(Pool *p)
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
void freeAllTensors(Pool *tensor_pool) {
    printf("freeing Tensors...\n");
    // Free the memory of each Tensor in the pool
    for (uint32_t i = 0; i < tensor_pool->blockCount; ++i) {
        if (tensor_pool->blocks[i] != NULL) {
            // Iterate over each Tensor in the block (blocksize is Tensor size)
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

        total_deallocated += sizeof(tensor_pool->blocks);

        tensor_pool->blocks = NULL;
    }

    freeAllData(); // Free all Datas objects

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
            freeAllData();
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
        Pool* tensorPool = fetchPool(TENSOR);
        freeBlock(tensorPool, t);
        t = NULL;
    }
}

/*  -----------------------------------------------------------------------------*/
/*  Data Memory Managment                                                        */
/*  -----------------------------------------------------------------------------*/
void setupGlobalDataPtrArray(int initial_capacity) {
    global_data_ptr_array = (DataPtrArray*)malloc(sizeof(DataPtrArray));
    global_data_ptr_array->data_ptrs = (Data**)malloc(sizeof(Data*) * initial_capacity);
    global_data_ptr_array->count = 0;
    global_data_ptr_array->capacity = initial_capacity;
}

void addDataPtr(Data* data_ptr) {
    if (global_data_ptr_array->count == global_data_ptr_array->capacity) {
        global_data_ptr_array->capacity *= 2;
        global_data_ptr_array->data_ptrs = (Data**)realloc(global_data_ptr_array->data_ptrs, sizeof(Data*) * global_data_ptr_array->capacity);
    }
    global_data_ptr_array->data_ptrs[global_data_ptr_array->count++] = data_ptr;
}
void freeAllData() {
    printf("freeing Datas...\n");
    if (global_data_ptr_array != NULL) {
        for (int i = 0; i < global_data_ptr_array->count; i++) {
            if (global_data_ptr_array->data_ptrs[i] != NULL) {
                // Free the memory allocated for the 'values' field
                if (global_data_ptr_array->data_ptrs[i]->values != NULL) {
                    free(global_data_ptr_array->data_ptrs[i]->values);
                    global_data_ptr_array->data_ptrs[i]->values = NULL;
                }
                // Free the memory allocated for the 'shape' field
                if (global_data_ptr_array->data_ptrs[i]->shape != NULL) {
                    free(global_data_ptr_array->data_ptrs[i]->shape);
                    global_data_ptr_array->data_ptrs[i]->shape = NULL;
                }
                // Free the memory allocated for the Data object itself
                free(global_data_ptr_array->data_ptrs[i]);
                global_data_ptr_array->data_ptrs[i] = NULL;
            }
        }
        // Free the memory allocated for the array of Data pointers
        free(global_data_ptr_array->data_ptrs);
        global_data_ptr_array->data_ptrs = NULL;
        // Free the memory allocated for the DataPtrArray object itself
        free(global_data_ptr_array);
        global_data_ptr_array = NULL;
    }
}