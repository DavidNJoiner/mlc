#ifndef MEMORY_POOL_H_ 
#define MEMORY_POOL_H_


/*  Simple memory pool manager following the LIFO approach for freeing and allocating memory.  */
/*  can help improve performance by reducing memory fragmentation.                        		*/

#include <stdint.h>
#include "config.h"
#include "tensor.h"
#include "device.h"
#include "data.h"
#include "function.h"
#include "nn.h"

#define POOL_BLOCKS_INITIAL 1
#define MAX_NUM_OBJECT_TYPES 8

#ifndef max
#define max(a,b) ((a)<(b)?(b):(a))
#endif

typedef struct PoolFreed {
	struct PoolFreed *nextFree;
} PoolFreed;

typedef struct {
	uint32_t elementSize;
	uint32_t blockSize;
	uint32_t used;
	int32_t block;
	PoolFreed *freed;
	uint32_t blocksUsed;
	uint8_t **blocks;
    PoolFreed *freedLast;
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

#ifndef DISABLE_MEMORY_POOLING

void *PoolMalloc(Pool *p);
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

#include <string.h>
#include <stdlib.h>
#include "memory_pool.h"

static GlobalPool global_pool_instance = {0};

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

void InitializeGlobalPool() {
    PoolInitialize(&global_pool_instance.pools[TENSOR], sizeof(Tensor), 256);
	PoolInitialize(&global_pool_instance.pools[DEVICE], sizeof(Device), 64);				//8 bytes  * 8 = 64 bits/obj
    PoolInitialize(&global_pool_instance.pools[DATA], sizeof(Data), 224);					//28 bytes * 8 = 224 bits/obj
    PoolInitialize(&global_pool_instance.pools[FUNCTION], sizeof(Function), 384);			//48 bytes * 8 = 384 bits/obj

	// Should have dedicated pool for Network objects
	PoolInitialize(&global_pool_instance.pools[NEURON], sizeof(Neuron), 128);				//16 bytes * 8 = 128 bits/obj
	PoolInitialize(&global_pool_instance.pools[LAYER], sizeof(Layer), 128);					//16 bytes * 8 = 128 bits/obj
	PoolInitialize(&global_pool_instance.pools[NN], sizeof(NeuralNet), 128);				//16 bytes * 8 = 128 bits/obj
	PoolInitialize(&global_pool_instance.pools[PARAMETERS], sizeof(Parameters), 128);		//16 bytes * 8 = 128 bits/obj
    global_pool_instance.is_initialized = true;
}

void FreeGlobalPool() {
    for (uint32_t i = 0; i < MAX_NUM_OBJECT_TYPES; i++) {
        PoolFreePool(&global_pool_instance.pools[i]);
    }
    global_pool_instance.is_initialized = false;
}
/*  ------------------------------------------------------------------------------------------------------------------------*/
/*  PoolInitialize: Sets up the memory pool with the specified size for each element and number of elements in each block.  */
/*  ------------------------------------------------------------------------------------------------------------------------*/
void PoolInitialize(Pool *p, const uint32_t elementSize, const uint32_t blockSize)
{
    p->elementSize = max(elementSize, sizeof(PoolFreed));
    p->blockSize = blockSize;

    PoolFreeAll(p);

    p->blocksUsed = POOL_BLOCKS_INITIAL;
    p->blocks = malloc(p->blocksUsed * sizeof(uint8_t*));

    for(uint32_t i = 0; i < p->blocksUsed; ++i)
        p->blocks[i] = NULL;
}

/*  ------------------------------------------------------------------------------------------------------------------------*/
/*  PoolFreePool: Frees all the memory associated with the pool. Should be called when done with the pool.                  */
/*  ------------------------------------------------------------------------------------------------------------------------*/
void PoolFreePool(Pool *p)
{
	uint32_t i;
	for(i = 0; i < p->blocksUsed; ++i) {
		if(p->blocks[i] == NULL)
			break;
		else
			free(p->blocks[i]);
	}

	free(p->blocks);
}

#ifndef DISABLE_MEMORY_POOLING

/*  ------------------------------------------------------------------------------------------------------------------------*/
/*  PoolMalloc: allocate memory for a new object from the pool.                                                             */
/*  ------------------------------------------------------------------------------------------------------------------------*/
void *PoolMalloc(Pool *p)
{   
    // Checks if there are any freed blocks that can be reused.
	if(p->freed != NULL) {
        void *recycle = p->freed;
        p->freed = p->freed->nextFree;
        if (p->freed == NULL) {
            p->freedLast = NULL;
        }
        return recycle;
    }

    //If not, it either uses the next available space in the current block or allocates a new block
	if(++p->used == p->blockSize) {
		p->used = 0;
		if(++p->block == (int32_t)p->blocksUsed) {
			uint32_t i;

			p->blocksUsed <<= 1;
			p->blocks = realloc(p->blocks, sizeof(uint8_t*)* p->blocksUsed);

			for(i = p->blocksUsed >> 1; i < p->blocksUsed; ++i)
				p->blocks[i] = NULL;
		}

		if(p->blocks[p->block] == NULL)
			p->blocks[p->block] = malloc(p->elementSize * p->blockSize);
	}
	
	return p->blocks[p->block] + p->used * p->elementSize;
}
/*  ------------------------------------------------------------------------------------------------------------------------*/
/*  PoolFree: Returns a block of memory back to the pool so it can be reused.                                               */
/*  ------------------------------------------------------------------------------------------------------------------------*/
void PoolFree(Pool *p, void *ptr)
{
    //he freed block is added to the list of freed blocks
    if(p->freed == NULL) {
        p->freed = ptr;
        p->freedLast = ptr;
    }
    else {
        p->freedLast->nextFree = ptr;
        p->freedLast = ptr;
    }
    p->freedLast->nextFree = NULL;
}

#endif //DISABLE_MEMORY_POOLING

/*  ------------------------------------------------------------------------------------------------------------------------*/
/*  PoolFreeAll: Resets the pool, marking all blocks as free.                                                               */
/*  ------------------------------------------------------------------------------------------------------------------------*/

// Note : This does not actually free the memory associated with the pool. For that, we use PoolFreePool.
void PoolFreeAll(Pool *p)
{
	p->used = p->blockSize - 1;
	p->block = -1;
	p->freed = NULL;
    p->freedLast = NULL;
}


#endif //MEMORY_POOL_IMPLEMENTATION