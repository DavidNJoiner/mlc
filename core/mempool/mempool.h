#ifndef MEMPOOL_H_
#define MEMPOOL_H_

#include "../../table.h"
#include <assert.h>
#include "global.h"
#include "../../define.h"
#include "../../device.h"
#include "../../tensor.h"

typedef struct SubBlock SubBlock_t;
typedef struct MemoryBlock MemoryBlock_t;
typedef struct Pool Pool_t;
typedef struct GlobalPool GlobalPool_t;

typedef SubBlock_t *SubBlock_ptr;
typedef MemoryBlock_t *MemoryBlock_ptr;

struct SubBlock
{
    size_t m_size;
    uint32_t m_ID;
};

struct MemoryBlock
{
    SubBlock_ptr freelist[MAX_ORDER + 2];
    alignas(DEEPC_SIZE_OF_VOID_POINTER) uint8_t m_subblock_array[BLOCKSIZE - 96];
};

struct Pool
{
    uint32_t m_numOfBlocks;     // Num of blocks
    uint32_t m_sizeOfEachBlock; // Size of each block
    uint32_t m_numFreeBlocks;   // Num of remaining blocks
    uint32_t m_numInitialized;  // Num of initialized blocks
    MemoryBlock_ptr m_memStart; // Beginning of memory pool
    MemoryBlock_ptr m_next;     // Next available block
};

struct GlobalPool
{
    Pool_t m_pools[MAX_POOL_INSTANCES];
    bool m_is_initialized;
};

#ifndef DISABLE_MEMORY_POOLING

// Pool memory managment
static void init_pool_(Pool_t *pool, const size_t poolSize);
void setup_pool(uint8_t pool_instance_index, size_t pool_size);
void display_pool_stats(Pool_t *pool);
void destroy_pool(Pool_t *pool);
Pool_t *fetch_pool();

// MemoryBlock managment
void free_block(Pool_t *pool, MemoryBlock_ptr block);
void print_memblock_info(MemoryBlock_ptr memblock);
MemoryBlock_ptr block_alloc(Pool_t *p);

// SubBlock managment
SubBlock_t *subblock_malloc(uint32_t size, MemoryBlock_ptr MEMBLOCK);
void subblock_free_all(MemoryBlock_ptr MEMBLOCK);
void remove_subblock(MemoryBlock_ptr memblock, SubBlock_ptr subblock);
void merge_subblocks(MemoryBlock_ptr memblock, SubBlock_ptr subblock1, SubBlock_ptr subblock2);
void optimize_layout(MemoryBlock_ptr memblock);

#else

#include <stdlib.h>
#define pool_malloc(p) malloc((p)->block_size)
#define pool_free(p, d) free(d)

#endif // DISABLE_MEMORY_POOLING

// Layer memory managment

// Neuron memory managment

// Tensor memory managment

// Data memory managment
void setup_global_data_ptr_array(int initial_capacity);
void add_data_ptr(Data *data_ptr);
void free_all_data();

// Debug functions
void print_list_subblock(MemoryBlock_ptr memblock, uint32_t i);
uint32_t count_blocks(uint32_t i);
uint32_t total_free();

#endif // MEMPOOL_H_
