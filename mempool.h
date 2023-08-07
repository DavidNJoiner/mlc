
#ifndef MEMPOOL_H_ 
#define MEMPOOL_H_

#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "config.h"
#include "device.h"
#include "tensor.h"

#define INITIAL_POOL_BLOCKS 1
#define MAX_OBJECT_TYPES 8

#ifndef max
#define max(a,b) ((a)<(b)?(b):(a))
#endif

typedef struct {
    size_t size;
    void* ptr;
} MemoryBlock;

typedef struct PoolFreed {
	struct PoolFreed* next_free;
} PoolFreed;

/* ---------------------------------------------------------------------------------------------- /
Pool:

element_size :           Size of each individual element within a block.
block_size :             Total size of each block of memory in the pool.
blocks_in_use :           Keeps track of the number of blocks that are currently in use within the current memory pool.
block :                 The index of the current block being used within the blocks array.
freed :                 Pointer to the list of freed memory blocks that are available for reuse.
blockCount:             Total number of memory blocks that have been allocated so far.
blocks :                Array of pointers to allocated blocks of memory.
freed_last :             Pointer to the last freed memory block in the list of freed blocks. 
                        Used for efficient addition of new freed blocks to the list.

total_obj_allocated :       Monitoring field that keeps track of the total number of objects that have been allocated.
total_recycled_blocks :     Monitoring field that keeps track of the total number of objects that have been recycled.
total_block_freed :         Monitoring field that keeps track of the total number of objects that have been deallocated.
new_block_allocations :   Monitoring field that keeps track of the total number of times a new block has been allocated.

/ ---------------------------------------------------------------------------------------------- */

typedef struct {
    ObjectType type;
    uint32_t element_size;
    uint32_t block_size;
    uint32_t blocks_in_use;
    int32_t block;
    PoolFreed* freed;
    uint32_t block_count;
    uint8_t** blocks;
    PoolFreed* freed_last;
    uint32_t total_obj_allocated;
    uint32_t total_recycled_blocks;
    uint32_t total_block_freed;
    uint32_t new_block_allocations; 
} Pool;

typedef struct {
    Pool pools[MAX_OBJECT_TYPES];
    bool is_initialized;
} GlobalPool;

Pool* fetch_pool(ObjectType type);

// Pool memory managment
void initialize_pool(ObjectType type, Pool *p, const uint32_t obj_size, const uint32_t num_obj, const uint32_t obj_per_block);
void calculate_pool_stats(Pool *p, size_t* total_allocated, size_t* total_pool_size);
void display_pool_stats(Pool *p);
void free_all_blocks(Pool *p);

// Tensor memory managment
void setup_tensor_pool(int num_tensors);
void destroy_tensor_pool();
void destroy_pool(Pool *p);
void free_tensor(Tensor* t);
void free_all_tensors();

// Data memory managment
void setup_global_data_ptr_array(int initial_capacity);
void add_data_ptr(Data* data_ptr);
void free_all_data();

#ifndef DISABLE_MEMORY_POOLING

// Block memory managment
MemoryBlock* create_block(Pool* p, void* block_address);
MemoryBlock* allocate_block(Pool *p);
void free_block(Pool *p, void *ptr);

#else

#include <stdlib.h>
#define pool_malloc(p) malloc((p)->block_size)
#define pool_free(p, d) free(d)

#endif

void free_all_blocks(Pool *p);

#endif 
