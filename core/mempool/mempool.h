#ifndef MEMPOOL_H_
#define MEMPOOL_H_

#include <assert.h>
#include <stddef.h>
#include <limits.h>
#include "state_manager.h"
#include "../define.h"
#include "../device.h"
#include "../../tensor.h"
#include "../../logging/table_cmd.h"

#define PARENT_OF(child_ptr, parent_type, parent_member) \
    ((parent_type *)((char *)(child_ptr)-offsetof(parent_type, parent_member)))

typedef struct SubBlock SubBlock_t;
typedef struct MemoryBlock MemoryBlock_t;
typedef struct Pool Pool_t;
typedef struct GlobalPool GlobalPool_t;

//------------------------------------------------------------
// SubBlock
//------------------------------------------------------------
struct SubBlock
{
    size_t m_size;
    uint32_t m_ID;
};
//------------------------------------------------------------
// MemoryBlock
//------------------------------------------------------------
struct MemoryBlock
{
    SubBlock_t *freelist[MAX_ORDER + 2];
    alignas(DEEPC_SIZE_OF_VOID_POINTER) uint8_t m_subblock_array[BLOCKSIZE - ((MAX_ORDER + 2) * DEEPC_SIZE_OF_VOID_POINTER)];
};
//------------------------------------------------------------
// Pool
//
// 24 bytes on 32 bit systems
// 32 bytes on 64 bit systems
//------------------------------------------------------------
struct Pool
{
    uint32_t m_numOfBlocks;     // Num of blocks
    uint32_t m_sizeOfEachBlock; // Size of each block
    uint32_t m_numFreeBlocks;   // Num of remaining blocks
    uint32_t m_numInitialized;  // Num of initialized blocks
    MemoryBlock_t *m_memStart;  // Beginning of memory pool
    MemoryBlock_t *m_next;      // Next available block
};

struct GlobalPool
{
    Pool_t m_pools[MAX_POOL_INSTANCES];
    bool m_is_initialized;
};

#ifndef DISABLE_MEMORY_POOLING

// Low level Pool memory managment
static void _init_pool_(Pool_t *pool, const size_t poolSize);
void init_pool(uint8_t pool_instance_index, size_t pool_size);
void display_pool_stats(Pool_t *pool);
void destroy_pool(Pool_t *pool);
Pool_t *fetch_pool();

// Low level MemoryBlock managment
MemoryBlock_t *memblock_alloc(Pool_t *p);
void block_free(Pool_t *pool, MemoryBlock_t *block);
void memblock_free(Pool_t *pool, MemoryBlock_t *block);

// Low level SubBlock managment
SubBlock_t *subblock_alloc(uint32_t size, MemoryBlock_t *MEMBLOCK);
void _subblock_free_(MemoryBlock_t *memblock, SubBlock_t *subblock);
void subblock_free_all(MemoryBlock_t *MEMBLOCK);
void _subblock_merge_(MemoryBlock_t *memblock, SubBlock_t *subblock1, SubBlock_t *subblock2);
void _subblock_coalescing_(MemoryBlock_t *memblock);

#else

#include <stdlib.h>
#define pool_malloc(p) malloc((p)->block_size)
#define pool_free(p, d) free(d)

#endif // DISABLE_MEMORY_POOLING

// Memory 
void* memory_alloc_padded (int size, int dtype);

// Layer memory managment

// Neuron memory managment

// Tensor memory managment

// Data memory managment
Data *data_alloc();
void setup_global_data_ptr_array(int initial_capacity);
void add_data_ptr(Data *data_ptr);
void free_all_data();

// Debug functions
void print_list_subblock(MemoryBlock_t *memblock, uint32_t i);
uint32_t count_free_pool_memoryblocks(uint32_t i);
uint32_t count_free_pool_bytes();

#endif // MEMPOOL_H_

#ifndef _MEMPOOL_IMPLEMENTATION_

void* memory_alloc_padded (int size, int dtype)
{
    int alignment_size = DEEPC_CPU;
    size_t element_size = sizeof(dtype);
    size_t padded_size = size * element_size;

    if (padded_size % alignment_size != 0) {
        size_t padding = alignment_size - (padded_size % alignment_size);
        padded_size += padding;
    }

    void *allocated_memory = malloc(padded_size);
    if (!allocated_memory) {
        return NULL;
    }
}

Data* data_alloc(){
    Data* data = malloc(sizeof(Data));
    if (!data) {
        perror("Error allocating Data structure");
        exit(EXIT_FAILURE);
    }
    return data;
}

#define _MEMPOOL_IMPLEMENTATION_

#endif // _IMPLEMENTATION_MEMPOOL_H_
