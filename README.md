## DEEPC : A Machine learning framework written in C for learning purpose.
### Memory Managment:
#### Expected Behavior for Tensor creation and destruction:

When a `Tensor` is instanciated. The programm looks for an available `MemoryBlocks` in a `Pool` instance.

| dtype                  | x64 byte size | x86 byte size |
| ---------------------- | ------------- | ------------- |
| struct SubBlock        | 8 bytes       | 8 bytes       |
| struct MemoryBlock     | 48 bytes      | 36 bytes      |
| struct Pool            | 28 bytes      | 28 bytes      |
| struct GlobalPool      | variable      | variable      |
| struct Tensor          | 16 bytes      | 16 bytes      |
| struct Data            | 20 bytes      | 20 bytes      |
