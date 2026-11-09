# Tests

Tests are added alongside each implementation stage. `test_scalar.py` already checks sigmoid values, numerical stability, and its local backward pass.

Planned organization:

- `test_scalar.py` — scalar expressions, the chain rule, and gradient accumulation.
- `test_tensor.py` — shapes, arithmetic, broadcasting, and indexing.
- `test_autograd.py` — backward passes and numerical gradient checks.
- `nn/` — layers, losses, CNNs, RNNs, LSTMs, GRUs, and attention.
- `optim/` — parameter updates and optimizer state.
- `data/` — batching, shuffling, and reproducibility.
- `test_serialization.py` — saving and restoring state.
- `test_generation.py` — sampling and matching logits with and without KV-cache.
- `reference/` — comparisons with PyTorch; these tests require the `reference` dependency group.

Use `pytest.importorskip("torch")` in reference tests so basic checks can run without PyTorch.

Check gradients on small `float64` arrays using finite differences. Use meaningful absolute and relative tolerances when comparing results.
