#ifndef MEMPOOL_H_
#define MEMPOOL_H_

#include <assert.h>
#include "define.h"
#include "device.h"
#include "tensor.h"

typedef struct SubBlock
{
    size_t m_SubBlockSize;
    uint32_t m_SubBlockID;
} SubBlock_t;

typedef struct MemoryBlock
{
    pointer_t freelist[MAX_ORDER + 2];
    uint8_t m_subblock_array[BLOCKSIZE];
} MemoryBlock_t;

typedef struct PoolFreed
{
    struct PoolFreed *m_nextFree;
} PoolFreed;

typedef struct Pool
{
    uint8_t **m_blocks;
    PoolFreed *m_freed;
    PoolFreed *m_freedLast;

    pointer_t m_next; // next free MemoryBlock
    pointer_t m_memStart;
    uint32_t m_numOfBlocks;
    uint32_t m_numFreeBlocks;
    uint32_t m_numInitialized;
    uint32_t m_sizeOfEachBlock;
} Pool_t;

typedef struct GlobalPool
{
    Pool_t m_pools[MAX_POOL_INSTANCES];
    bool m_is_initialized;
} GlobalPool_t;

// Pool memory managment
static void init_pool_(Pool_t *pool, const uint32_t poolSize);
void setup_pool(uint8_t poolInstanceID, uint32_t poolByteSize);
void display_pool_stats(Pool_t *pool);
void destroy_pool(Pool_t *pool);
MemoryBlock_t *create_block(Pool_t *pool);
MemoryBlock_t *block_alloc(Pool_t *p);
Pool_t *fetch_pool();

// Layer memory managment

// Neuron memory managment

// Tensor memory managment

// Data memory managment
void setup_global_data_ptr_array(int initial_capacity);
void add_data_ptr(Data *data_ptr);
void free_all_data();

#ifndef DISABLE_MEMORY_POOLING

// MemoryBlock managment
void memblock_init(MemoryBlock_t *memblock);
void memblock_deinit(MemoryBlock_t *memblock);
void free_block(Pool_t *pool, MemoryBlock_t *block);
void print_memblock(MemoryBlock_t *memblock);

// SubBlock managment
pointer_t subblock_malloc(uint32_t size, MemoryBlock_t *MEMBLOCK);
void subblock_free_all(MemoryBlock_t *MEMBLOCK);

#else

#include <stdlib.h>
#define pool_malloc(p) malloc((p)->block_size)
#define pool_free(p, d) free(d)

#endif

#endif
