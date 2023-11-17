#ifndef MEMPOOL_H_
#define MEMPOOL_H_

/*
 If your header file is included in multiple translation units (source files), make sure to handle memory management cautiously,
 especially with functions that perform memory allocation. If possible, keep allocation and deallocation within the same
 translation unit to avoid potential issues.
*/

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
typedef struct MemBlock MemBlock_t;
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
// MemBlock
//------------------------------------------------------------
struct MemBlock
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
    MemBlock_t *m_memStart;  // Beginning of memory pool
    MemBlock_t *m_next;      // Next available block
};

struct GlobalPool
{
    Pool_t m_pools[MAX_POOL_INSTANCES];
    bool m_is_initialized;
};

                // Low level Pool memory managment

Pool_t*         pool_get_from_index(int pool_index);
uint32_t        pool_count_free_bytes(int pool_index);

static void     pool_init_debug(Pool_t *pool, const size_t poolSize);
void            pool_init(uint8_t pool_instance_index, size_t pool_size);
void            pool_print_stats(Pool_t *pool);
void            pool_destroy(Pool_t *pool);

                // Low level MemBlock managment

uint32_t        memblock_count_free_subblocks(MemBlock_t* memblock_ptr);
MemBlock_t*     memblock_alloc(Pool_t *p);

                // TODO: clean this mess. block_free and memblock_free are comfusing !
void            block_free(Pool_t *pool, MemBlock_t *block); 
void            memblock_free(Pool_t *pool, MemBlock_t *block);

                // Low level SubBlock managment

SubBlock_t*     subblock_alloc(uint32_t size, MemBlock_t *MEMBLOCK);
void            _subblock_free_(MemBlock_t *memblock, SubBlock_t *subblock);
void            subblock_free_all(MemBlock_t *MEMBLOCK);
void            _subblock_merge_(MemBlock_t *memblock, SubBlock_t *subblock1, SubBlock_t *subblock2);
void            _subblock_coalescing_(MemBlock_t *memblock);


// Memory 
void* memory_alloc_padded (int size, int dtype);

// Layer memory managment

// Neuron memory managment

// Tensor memory managment

// Array memory managment
arr_t *arr_alloc();
void arr_init_global_ptr_count(int initial_capacity);
void arr_increment_ptr_count(arr_t *data_ptr);
void arr_free_all();

// Debug functions
void subblock_print(MemBlock_t *memblock, uint32_t i);

#endif // MEMPOOL_H_

