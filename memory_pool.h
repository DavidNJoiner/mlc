#ifndef MEMORY_POOL_H_ 
#define MEMORY_POOL_H_

#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include "config.h"
#include "device.h"

#define POOL_BLOCKS_INITIAL 1
#define MAX_NUM_OBJECT_TYPES 8

#ifndef max
#define max(a,b) ((a)<(b)?(b):(a))
#endif

typedef struct {
    size_t size;
    void* ptr;
} MemoryBlock;

typedef struct PoolFreed {
	struct PoolFreed* nextFree;
} PoolFreed;

typedef struct {
    uint32_t elementSize;
    uint32_t blockSize;
    uint32_t used;
    int32_t block;
    PoolFreed* freed; // Freed memory blocks available
    uint32_t blocksUsed;
    uint8_t** blocks;
    PoolFreed* freedLast;
    // Monitoring field
    uint32_t numAllocated;  // Number of allocated objects
    uint32_t numDeallocated;  // Number of deallocated objects
    uint32_t numBlockAllocations;  // Number of block allocations
} Pool;

typedef struct {
    Pool pools[MAX_NUM_OBJECT_TYPES];
    bool is_initialized;
} GlobalPool;

Pool* GetPool(ObjectType type);

void InitializeGlobalPool();
void FreeGlobalPool();

void PoolInitialize(Pool *p, const uint32_t elementSize, const uint32_t blockSize);
void PoolFreePool(Pool *p);
void freeTensor(Tensor* t);
void PoolDeepFree(Pool *p);

#ifndef DISABLE_MEMORY_POOLING

MemoryBlock* PoolMalloc(Pool *p);
void PoolFree(Pool *p, void *ptr);

#else

#include <stdlib.h>
#define PoolMalloc(p) malloc((p)->blockSize)
#define PoolFree(p, d) free(d)

#endif

void PoolFreeAll(Pool *p);


#endif //MEMORY_POOL_H_





#ifndef MEMORY_POOL_IMPLEMENTATION
#define MEMORY_POOL_IMPLEMENTATION

#include "memory_pool.h"
#include "tensor.h"

static GlobalPool global_pool_instance = {0};


/* ------------------------------------------------------------------------------------------------ */
/* PoolReportStats: Print statistics about the pool to the console.
 *
 * p: Pointer to the Pool for which statistics are being reported.
 */
/* ------------------------------------------------------------------------------------------------ */
void PoolReportStats(Pool *p)
{
    printf("Allocated objects: %u\n", p->numAllocated);
    printf("Deallocated objects: %u\n", p->numDeallocated);
    printf("Block allocations: %u\n", p->numBlockAllocations);
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
        InitializeGlobalPool();
    }
    // Return the common pool for any Function subclass
    if (type >= FUNCTION && type <= LAST_FUNCTION_SUBCLASS) {
        return &global_pool_instance.pools[FUNCTION];
    }
    // For non-Function types, return their specific pool
    return &global_pool_instance.pools[type];
}

/* ------------------------------------------------------------------------------------------------ */
/* InitializeGlobalPool: Initialize the global memory pool with specific pools for each ObjectType.
 */
/* ------------------------------------------------------------------------------------------------ */
void InitializeGlobalPool() {
    PoolInitialize(&global_pool_instance.pools[TENSOR], sizeof(Tensor), 1024);
 /*    PoolInitialize(&global_pool_instance.pools[DEVICE], sizeof(Device), 16);			
    PoolInitialize(&global_pool_instance.pools[DATA], sizeof(Data), 16);					
    PoolInitialize(&global_pool_instance.pools[FUNCTION], sizeof(Function), 2048);	 */		

    // Should have a dedicated pool for Network objects
   /*  PoolInitialize(&global_pool_instance.pools[NEURON], sizeof(Neuron), 512);				
    PoolInitialize(&global_pool_instance.pools[LAYER], sizeof(Layer), 8);			
    PoolInitialize(&global_pool_instance.pools[NN], sizeof(NeuralNet), 4);				
    PoolInitialize(&global_pool_instance.pools[PARAMETERS], sizeof(Parameters), 512);	 */	
    global_pool_instance.is_initialized = true;
}

/* ------------------------------------------------------------------------------------------------ */
/* FreeGlobalPool: Free the memory of all pools in the global pool instance and print statistics.
 */
/* ------------------------------------------------------------------------------------------------ */
void FreeGlobalPool() {
    for (uint32_t i = 0; i < MAX_NUM_OBJECT_TYPES; i++) {
        PoolReportStats(&global_pool_instance.pools[i]);
        PoolFreePool(&global_pool_instance.pools[i]);
    }
    global_pool_instance.is_initialized = false;
}

/* ---------------------------------------------------------------------------------------------- */
/* PoolInitialize: Initialize a memory pool with the specified characteristics.
 *
 * p: Pointer to the Pool for which memory is being allocated.
 * elementSize: The size of one object.
 * blockSize: The number of object instances that can fit in one block.
 */
/* ---------------------------------------------------------------------------------------------- */
void PoolInitialize(Pool *p, const uint32_t elementSize, const uint32_t blockSize)
{
    // Set the elementSize to the maximum of specified size and PoolFreed size.
    p->elementSize = max(elementSize, sizeof(PoolFreed));

    // Set the blockSize.jhjh
    p->blockSize = blockSize;

    // Free all previously allocated memory.
    PoolFreeAll(p);

    // Allocate memory for blocks.
    p->blocksUsed = POOL_BLOCKS_INITIAL;
    p->blocks = malloc(p->blocksUsed * sizeof(uint8_t*));

    // Initialize blocks to NULL.
    for (uint32_t i = 0; i < p->blocksUsed; ++i)
        p->blocks[i] = NULL;

    // Reset the monitoring fields.
    p->numAllocated = 0;
    p->numDeallocated = 0;
    p->numBlockAllocations = 0;
}


/* ------------------------------------------------------------------------------------------------ */
/* PoolFreePool: Frees all the memory associated with the pool. Should be called when done with the pool.
 *
 * p: Pointer to the Pool to be freed.
 */
/* ------------------------------------------------------------------------------------------------ */
void PoolFreePool(Pool *p)
{
    uint32_t i;
    for (i = 0; i < p->blocksUsed; ++i) {
        if (p->blocks[i] == NULL)
            break;
        else
            free(p->blocks[i]);
    }

    free(p->blocks);
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
    p->numAllocated++;

    if (p->freed != NULL) {
        MemoryBlock* recycle = (MemoryBlock*)p->freed;
        p->freed = p->freed->nextFree;
        if (p->freed == NULL) {
            p->freedLast = NULL;
        }
        return recycle;
    }

    if (++p->used == p->blockSize) {
        p->used = 0;
        if (++p->block == (int32_t)p->blocksUsed) {
            uint32_t i;

            p->blocksUsed <<= 1;
            p->blocks = realloc(p->blocks, sizeof(uint8_t*) * p->blocksUsed);

            for (i = p->blocksUsed >> 1; i < p->blocksUsed; ++i)
                p->blocks[i] = NULL;
        }

        if (p->blocks[p->block] == NULL) {
            // Increment the number of block allocations
            p->numBlockAllocations++;
            p->blocks[p->block] = malloc(p->elementSize * p->blockSize);
        }
    }
    
    MemoryBlock* block = (MemoryBlock*)(p->blocks[p->block] + p->used * p->elementSize);
    block->size = p->elementSize;
    block->ptr = NULL;
    return block;
}

/* ------------------------------------------------------------------------------------------------ */
/* PoolFree: Returns a block of memory back to the specified pool so it can be reused.
 *
 * p: Pointer to the Pool from which to free the memory.
 * ptr: Pointer previously returned by PoolMalloc which needs to deallocate that specific memory block.
 */
/* ------------------------------------------------------------------------------------------------ */
void PoolFree(Pool *p, void *ptr)
{
    // Increment the number of deallocated objects
    p->numDeallocated++;

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
/* PoolFreeAll: Resets the pool, marking all blocks as free.
 * Note: This does not actually free the memory associated with the pool. For that, use PoolFreePool.
 *
 * p: Pointer to the Pool to be reset.
 */
/* ------------------------------------------------------------------------------------------------ */
void PoolFreeAll(Pool *p)
{
    p->used = p->blockSize - 1;
    p->block = -1;
    p->freed = NULL;
    p->freedLast = NULL;
}

/* ------------------------------------------------------------------------------------------------ */
/* PoolDeepFreeTensor: Frees all the memory associated with a given Tensor pool, including the      */
/* memory of the Tensors stored in the pool. This function should be used with caution, as it will  */
/* invalidate all Tensors that were allocated from the pool. After calling this function, the pool  */
/* can be reused for new allocations.                                                               */
/*                                                                                                  */
/* p: Pointer to the Pool to be deeply freed.                                                       */
/* ------------------------------------------------------------------------------------------------ */
void PoolDeepFree(Pool *p) {
    // Free the memory of each Tensor in the pool
    for (uint32_t i = 0; i < p->blocksUsed; ++i) {
        if (p->blocks[i] != NULL) {
            // Iterate over each Tensor in the block
            for (uint32_t j = 0; j < p->blockSize; ++j) {
                // Calculate the address of the Tensor
                Tensor* tensor_ptr = (Tensor*)(p->blocks[i] + j * p->elementSize);
                // Free the Tensor
                freeTensor(tensor_ptr);
            }
            // Free the block itself
            free(p->blocks[i]);
        }
    }

    // Free the array of blocks
    free(p->blocks);

    // Reset the pool
    p->used = p->blockSize - 1;
    p->block = -1;
    p->freed = NULL;
    p->freedLast = NULL;
}
/*  ---------------------------------------------------------------*/
/*  freeTensor : Releases the memory allocated for a given tensor. */
/*  ---------------------------------------------------------------*/
void freeTensor(Tensor* t) {
    if (t != NULL) {
        if (t->data != NULL) {
            if (t->data->values != NULL) {
                if (t->device->type == CUDA) {
                    cudaFree(t->data->values); 
                } else {
                    free(t->data->values);  
                }
                t->data->values = NULL;
            }
            // Data objects are not (yet?) allocated using the memory pool, we use free instead of PoolFree
            free(t->data);
            t->data = NULL;
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
        PoolFree(tensorPool, t);
        t = NULL;
    }
}


#endif //MEMORY_POOL_IMPLEMENTATION