# **mlc** (project deepc)
Machine learning framework written in C for learning purpose

### todo:

Arithmetic

- [ ] arithmetic ops for all dtypes.
- [ ] **cpu_ops** : use intrinsics for differents architecture.
- [ ] **gpu_ops** : CUDA ops.

Tensor
- [ ] function to create tensor in place. no Data object required.
- [x] function to create empty tensor with required shape and dimension.
- [x] function to create empty tensor from existing Tensor.

Debug

- [ ] implement a **logger**.
- [x] print tensor should respect the shape of the tensor.

ML

- [ ] implement simple **neuron**.
- [ ] implement simple **layer**.
- [ ] implement simple **nn**.
- [ ] autograd.
