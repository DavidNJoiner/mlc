# **mlc** (project deepc)
Machine learning framework written in C for learning purpose. 
---

![deepc_logo](https://github.com/DavidNJoiner/mlc/assets/69796012/d1d7daee-5789-4464-bcff-1b845a91ac27)

---
### todo:

Arithmetic

- [ ] Arithmetic ops for all dtypes.
- [ ] **Intel AVX/AVX2 ops** : Intrinsics ops + GEMM.
- [ ] **Nvidia CUDA ops**.
- [ ] **AMD mroc ops**.
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

- [ ] Implement a simple **neuron** struct.
- [ ] Implement simple **layer** struct.
- [ ] Implement simple **nn** struct.
- [ ] Implement activation functions and their derivatives.
- [ ] Autograd engine.
- [ ] Adam optim.

---
### goals:

- [ ] Run stable diffusion.
- [ ] Any other fun stuff !
