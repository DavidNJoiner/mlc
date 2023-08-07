## DEEPC : A Machine learning framework written in C for learning purpose.

### Memory Managment:

## MLC : A deep learning framework written in C

Expected Behavior:

A `Tensor` is allocated -> The allocator looks in the TensorPool's `blocks` for free `MemoryBlock`.

The number `Tensor` that can be held by a single `MemoryBlock`is controled by the constant @@MAX_OBJ_PER_BLOCK.
If it can't find one, it will allocate a new `block` pointer in `blocks`and create a new `MemoryBlock` at that address.

When a `Tensor` is de-allocated, the memory isn't returned to the system directly. 
Instead, the space is recycled in the TensorPool's available memory for future use. 
This is because the allocation and de-allocation of memory are expensive operations in terms of time.              

Explanation:

- A memory pool consists of multiple blocks, each of which is a contiguous section of memory.
- Each block is divided into smaller elements, and all elements have the same size.
- The objects are allocated from the elements within the blocks.
- An allocated object occupies one element within a block.
- When an object is deallocated, the element becomes available for reuse.
- The memory pool manages the allocation and deallocation of objects within the blocks.
- The `used` field in the `Pool` struct keeps track of the number of used elements within the current block.
- When the current `MemoryBlock` is fully utilized (i.e., `used == block_size`) a new block is allocated.
- The memory pool may grow dynamically by adding more blocks as needed.
- The `blocks` array holds the pointers to the allocated `MemoryBlock`.
