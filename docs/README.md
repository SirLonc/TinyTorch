# Implementation notes

As the project progresses, this directory will contain explanations of the mathematics, tensor shapes, and implementation decisions.

For each component, document:

1. Its mathematical definition and input/output shapes.
2. The forward and backward passes.
3. Edge cases and numerical stability.
4. Correctness checks and known limitations.

Component dependencies:

- `tensor` and `autograd` form the computational foundation; they do not depend on layers or optimizers.
- `nn` builds operations and layers on tensors. Layer backward passes are composed from differentiable operations where possible.
- `optim` updates parameters using their gradients.
- `data` handles dataset access and batch construction.
- `generation` uses trained models for autoregressive inference.
- `examples` combine library components into specific models and experiments.

The initial scalar `Value` stage is a separate educational prototype. `Tensor` should not store arrays of `Value` objects: tensor autograd will operate on whole arrays.
