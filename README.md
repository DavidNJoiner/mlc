# **mlc** (project deepc)
## Machine learning framework written in C for learning purpose.
---

![deepc_logo](https://github.com/DavidNJoiner/mlc/assets/69796012/d1d7daee-5789-4464-bcff-1b845a91ac27)

---
### TODO:

Arithmetic

- [ ] Arithmetic ops for all usual datatypes.
- [x] **Intel AVX/AVX2 ops** : Intrinsics ops + GEMM for fp32 and fp64.
- [x] **Nvidia CUDA ops** : For fp16.
- [ ] **cuBLAS ops**
- [ ] **Open CL ops**.

Tensor

- [x] Function to create an empty *Tensor* with desired shape and dimension. No *Data* object required.
- [x] Function to create "zero" *Tensor* from an existing Tensor.
- [ ] Function to fully transpose a *Tensors*.
- [ ] Function to normalize a *Tensors*'s data.

Debug

- [ ] Implement a simple extensible **logger**.
- [x] *printTensor* should respect the shape of *Tensor*'s data..
- [x] *printTensor* should work for any dtype and any data dimension.

ML

- [ ] Implement a **neuron** class.
- [ ] Implement a **layer** class.
- [ ] Implement a **nn** class.
- [ ] Implement a **Function** class.
- [ ] Implement an Automatic differentiation engine (Autograd).
- [ ] Implement an **Optimizer** class + SGD, AdamW.

---
### Memory Managment:

/*---------------------------------------------------------------------------------------------------------------------*/
/*                                             Caching allocators                                                      */
/*---------------------------------------------------------------------------------------------------------------------*/

    EXPECTED BEHAVIOR 
    
    When a tensor is allocated, the allocator looks in the Pool for a free block that is closest in size
    to the requested size. If it can't find one, it will allocate a new block.
    When a tensor is de-allocated, the memory isn't returned to the system directly. 
    Instead, the space is kept in a pool of available memory blocks for future use. 
    This is because the allocation and de-allocation of memory are expensive operations in terms of time.              

+-----------------------+
|      Memory Pool      |
+-----------------------+
|                       |
|         Blocks        |
|                       |
+-----------------------+
|       |         |     |
| Block |  Block  | ... |
|   0   |    1    |     |
|       |         |     |
+-------+---------+-----+
|                       |
|        Objects        |
|                       |
+-----------------------+
| Object | Object | ... |
|   0    |   1    |     |
|        |        |     |
+--------+--------+-----+

- Memory Pool: The overall memory pool structure.
- Blocks: Blocks within the memory pool.
- Objects: Allocated objects within the blocks.

Explanation:
- A memory pool consists of multiple blocks, each of which is a contiguous section of memory.
- Each block is divided into smaller elements, and all elements have the same size.
- The objects are allocated from the elements within the blocks.
- An allocated object occupies one element within a block.
- When an object is deallocated, the element becomes available for reuse.
- The memory pool manages the allocation and deallocation of objects within the blocks.
- The `used` field in the `Pool` struct keeps track of the number of used elements within the current block.
- When the current block is fully utilized (i.e., `used == blockSize`), a new block is allocated.
- The memory pool may grow dynamically by adding more blocks as needed.
- The `blocks` array holds the pointers to the allocated blocks.
- Each block can be represented as an array of elements, where each element can store an object.
