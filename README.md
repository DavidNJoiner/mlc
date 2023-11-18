## DEEPC : A Machine learning framework written in C for learning purpose.
### Memory Managment:


### Structure and Hierarchy:

 DISCLAIMER : *The current memory allocation system is very much a work in progress. It is an hybrid solution which i haven't tested yet so i have no idea how such fragmentation will affect the efficiency in long-running systems.*

+ **Pool Structure:** The Pool structure holds essential metadata regarding the memory pool, such as block size, number of blocks, initialization status, etc. This structure is based upon the paper [*"Fast Efficient Fixed-Size Memory Pool"*](https://arxiv.org/ftp/arxiv/papers/2210/2210.16471.pdf) by Ben Kenwright.

+ **MemBlock:** This structure utilizes the buddy allocation scheme with a free list for each order. 

+ **SubBlock:** This is a further partitioning of the buddy allocator, SubBlock structure carries size and ID information. To offers an additional level of fine-grained control over memory allocation.


### Expected Behavior for Tensor creation and destruction:

When a `Tensor` is instanciated. The programm looks for an available `MemoryBlocks` in a `Pool` instance.


### Ops Lookup:

The filled cells represent native numerical type support, while the goal is to ensure compatibility for the remaining cells through software solutions.

|         | DTYPE      | quint4x2 | quint8 | qint8 | int8 | uint8 | int16 | uint16 | int32 | uint32 | int64 | float16 | bfloat16 | float32 | float64 | bool |
|:-------:|:----------:|:--------:|:------:|:-----:|:----:|:-----:|:-----:|:------:|:-----:|:------:|:-----:|:-------:|:--------:|:-------:|:-------:|:----:|
| **Intel CPU** | **AMD CPU** |          |        |       |      |       |       |        |       |        |       |         |          |         |         |      |
| AVX512  | AVX-512 FMA|          |        |       | x    | x     | x     | x      | x     | x      | x     | x       |          | x       | x       | x    |
| AVX2    | AVX2       |          |        |       | x    | x     | x     | x      | x     | x      | x     |         |          | x       | x       | x    |
| AVX     | AVX        |          |        |       | x    | x     | x     | x      | x     | x      | x     |         |          | x       | x       | x    |
| SSE4.2  | SSE4a      |          |        |       | x    | x     | x     | x      | x     | x      | x     |         |          | x       | x       | x    |
| SSE4.1  | SSE4.1     |          |        |       | x    | x     | x     | x      | x     | x      | x     |         |          | x       | x       | x    |
| SSSE3   | SSSE3      |          |        |       | x    | x     | x     | x      | x     | x      | x     |         |          | x       | x       | x    |
| SSE3    | SSE3       |          |        |       | x    | x     | x     | x      | x     | x      | x     |         |          | x       | x       | x    |
| SSE2    | SSE2       |          |        |       | x    | x     | x     | x      | x     | x      | x     |         |          | x       | x       | x    |
| SSE     | SSE        |          |        |       | x    | x     | x     | x      | x     | x      | x     |         |          | x       | x       | x    |

