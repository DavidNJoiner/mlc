## DEEPC : A Machine learning framework written in C for learning purpose.
### Memory Managment:


### Structure and Hierarchy:

 DISCLAIMER : *The current memory allocation system is very much a work in progress. It is an hybrid solution which i haven't tested yet so i have no idea how such fragmentation will affect the efficiency in long-running systems.*

+ **Pool Structure:** The Pool structure holds essential metadata regarding the memory pool, such as block size, number of blocks, initialization status, etc. This structure is based upon the paper [*"Fast Efficient Fixed-Size Memory Pool"*](https://arxiv.org/ftp/arxiv/papers/2210/2210.16471.pdf) by Ben Kenwright.

+ **MemBlock:** This structure utilizes the buddy allocation scheme with a free list for each order. 

+ **SubBlock:** This is a further partitioning of the buddy allocator, SubBlock structure carries size and ID information. To offers an additional level of fine-grained control over memory allocation.


### Expected Behavior for Tensor creation and destruction:

When a `Tensor` is instanciated. The programm looks for an available `MemoryBlocks` in a `Pool` instance.


### Common Object Sizes:

| dtype                  | x64 byte size | x86 byte size |
| ---------------------- | ------------- | ------------- |
| struct SubBlock        | 8 bytes       | 8 bytes       |
| struct MemBlock        | 48 bytes      | 36 bytes      |
| struct Pool            | 28 bytes      | 28 bytes      |
| struct GlobalPool      | variable      | variable      |
| struct Tensor          | 16 bytes      | 16 bytes      |
| struct arr_t           | 20 bytes      | 20 bytes      |
