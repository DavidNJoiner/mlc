# **mlc** (project deepc)
Machine learning framework written in C for learning purpose. 
---

![deepc_logo](https://github.com/DavidNJoiner/mlc/assets/69796012/46ec6565-f34d-43f9-a464-cb5a9a9d8218)

---
### todo:

Arithmetic

- [ ] arithmetic ops for all dtypes.
- [ ] **cpu_ops** : use intrinsics for differents architecture.
- [ ] **gpu_ops** : CUDA ops.

Tensor
- [x] function to create empty tensor with required shape and dimension. no Data object required.
- [x] function to create empty tensor from existing Tensor.
- [ ] function to transpose Tensors.

Debug

- [ ] implement a **logger**.
- [x] print tensor should respect the shape of the tensor.
- [x] print tensor with any dtype and any dimension.

ML

- [ ] implement simple **neuron**.
- [ ] implement simple **layer**.
- [ ] implement simple **nn**.
- [ ] implement activation functions and their derivatives.
- [ ] autograd engine.
- [ ] Adam optim.

---
### goals:

- [ ] Run stable diffusion.
- [ ] Any other fun stuff !
