#ifndef MEMPOOL_H_ 
#define MEMPOOL_H_

#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#include <math.h>

#include "config.h"
#include "device.h"
#include "tensor.h"

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

/* ---------------------------------------------------------------------------------------------- /
Pool:

elementSize :           Size of each individual element within a block.
blockSize :             Total size of each block of memory in the pool.
BlocksInUse :           Keeps track of the number of blocks that are currently in use within the current memory pool.
block :                 The index of the current block being used within the blocks array.
freed :                 Pointer to the list of freed memory blocks that are available for reuse.
blockCount:             Total number of memory blocks that have been allocated so far.
blocks :                Array of pointers to allocated blocks of memory.
freedLast :             Pointer to the last freed memory block in the list of freed blocks. 
                        Used for efficient addition of new freed blocks to the list.

totalObjAllocated :       Monitoring field that keeps track of the total number of objects that have been allocated.
totalRecycledBlocks :     Monitoring field that keeps track of the total number of objects that have been recycled.
totalBlockFreed :         Monitoring field that keeps track of the total number of objects that have been deallocated.
newBlockAllocations :   Monitoring field that keeps track of the total number of times a new block has been allocated.

/ ---------------------------------------------------------------------------------------------- */
typedef struct {
    ObjectType type;
    uint32_t elementSize;
    uint32_t blockSize;
    uint32_t BlocksInUse;
    int32_t block;
    PoolFreed* freed;
    uint32_t blockCount;
    uint8_t** blocks;
    PoolFreed* freedLast;
    uint32_t totalObjAllocated;
    uint32_t totalRecycledBlocks;
    uint32_t totalBlockFreed;
    uint32_t newBlockAllocations; 
} Pool;

typedef struct {
    Pool pools[MAX_NUM_OBJECT_TYPES];
    bool is_initialized;
} GlobalPool;

Pool* fetchPool(ObjectType type);

void initializePool(ObjectType type, Pool *p, const uint32_t obj_size, const uint32_t num_obj, const uint32_t obj_per_block);
void setupTensorPool(int nb_tensors);
void destroyTensorPool();
void destroyPool(Pool *p);
void freeTensor(Tensor* t);
void freeAllTensors(Pool *p);
void calculatePoolStats(Pool *p, size_t* total_allocated, size_t* total_pool_siz);
void displayPoolStats(Pool *p);
void freeAllBlocks(Pool *p);

//Data
void setupGlobalDataPtrArray(int initial_capacity);
void addDataPtr(Data* data_ptr);
void freeAllData();

#ifndef DISABLE_MEMORY_POOLING

MemoryBlock* createBlock(Pool* p, void* blockAddress);
MemoryBlock* allocateBlock(Pool *p);
void freeBlock(Pool *p, void *ptr);

#else

#include <stdlib.h>
#define PoolMalloc(p) malloc((p)->blockSize)
#define PoolFree(p, d) free(d)

#endif

void freeAllBlocks(Pool *p);


#endif //MEMPOOL_H_