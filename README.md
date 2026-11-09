# TinyTorch

An educational implementation of the core mechanisms of PyTorch, from tensors and automatic differentiation to training CNNs, LSTMs, GRUs, and Transformers.

The goal is to understand the mathematics and implementation of each component, build a framework from scratch, and train small models with it.

## Approach

- Python + NumPy: NumPy provides array storage and numerical operations; autograd, layers, and optimizers are implemented from scratch.
- PyTorch serves as a reference in tests for comparing outputs and gradients.
- Start with clear, straightforward implementations, then optimize after verifying correctness.
- All architectures share tensors, autograd, layer interfaces, and optimizers.

## Current status

The package skeleton and modules for all stages are in place. The scalar `Value` class is partially implemented, including a graph traversal; some behavior still needs further validation. Sigmoid has tests for values, numerical stability, and its local backward pass. `Tensor` remains a placeholder: constructing one raises `NotImplementedError`. Other modules describe their planned responsibilities; executable examples will be added alongside implementations.

## Installation and development

Python 3.11 or newer is required. We use [uv](https://docs.astral.sh/uv/guides/projects/) to manage the environment and dependencies.

`pyproject.toml` contains project metadata, dependency requirements, and tool settings. uv reads this file, creates `.venv`, and records exact dependency versions in `uv.lock`.

On macOS, install uv with Homebrew if you use it:

```bash
brew install uv
```

See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for other options.

From the project root, create the environment and install the package with development tools:

```bash
uv sync
```

The `dev` group (`pytest`, `ruff`) is installed by default. The `reference` group contains PyTorch and is enabled explicitly:

```bash
uv sync --group reference
```

Run commands with `uv run`, without manually activating the environment:

```bash
uv run python -c "import tinytorch; from tinytorch.autograd import Value"
uv run ruff check .
uv run ruff format --check .
```

Run tests with `uv run pytest`; run future PyTorch comparisons with `uv run --group reference pytest`.

Manage dependencies:

```bash
uv add <package>                    # Add a library dependency
uv add --dev <package>              # Add a development tool
uv add --group reference <package>  # Add a reference-testing dependency
uv remove <package>                 # Remove a library dependency
uv lock --upgrade-package numpy     # Update the locked NumPy version
```

Replace `<package>` with the package name. Commit the generated `uv.lock` alongside `pyproject.toml`. On other machines, use `uv sync --locked` to check that the lockfile is current and install the locked versions. Do not commit `.venv/`. See the [dependency documentation](https://docs.astral.sh/uv/concepts/projects/dependencies/) for details.

## Project structure

```text
TinyTorch/
├── pyproject.toml           # Installation, dependencies, pytest, and Ruff
├── src/tinytorch/
│   ├── tensor.py            # Tensors and operations
│   ├── autograd/            # Scalar prototype and tensor autograd
│   ├── nn/
│   │   ├── parameter.py     # Trainable parameters
│   │   ├── functional.py    # Functional operations
│   │   ├── init.py          # Weight initialization
│   │   ├── utils.py         # Gradient clipping
│   │   └── modules/         # Basic layers, CNN, RNN, and Transformer
│   ├── optim/               # SGD, Adam, and optimizer state
│   ├── data/                # Dataset and DataLoader
│   ├── serialization.py     # Saving and loading
│   ├── random.py            # Reproducibility
│   └── generation.py        # Sampling and KV-cache integration
├── tests/                   # Correctness checks and PyTorch comparisons
├── examples/                # Educational models and training
├── docs/                    # Mathematics and implementation decisions
└── benchmarks/              # Performance measurements
```

## Implementation roadmap

### 1. Scalar autograd

- [ ] `Value`: a scalar, its gradient, and links to previous operations.
- [ ] Addition, multiplication, powers, and basic activation functions.
- [ ] Computation graph, topological traversal, and `backward()`.
- [ ] Gradient accumulation when a value is reused.

**Outcome:** automatic differentiation of scalar expressions and an understanding of the chain rule.

### 2. Tensors

- [ ] `Tensor`: data, `shape`, `dtype`, `requires_grad`, and `grad`.
- [ ] Arithmetic operations and broadcasting.
- [ ] Indexing, `reshape`, transposition, and axis permutations.
- [ ] Reductions: `sum`, `mean`.
- [ ] Matrix and batched matrix multiplication.

**Outcome:** vectors, matrices, and batches through a consistent interface.

### 3. Tensor autograd

- [ ] Dynamic computation graph and backward pass for tensor operations.
- [ ] Gradients for broadcasting, reductions, and reshaping.
- [ ] Indexing gradients, including repeated indices.
- [ ] `backward()`, gradient resetting, and accumulation.
- [ ] `detach()` and the `no_grad()` context.

**Outcome:** parameter gradients for expressions such as `loss = ((x @ w + b - y) ** 2).mean()`.

### 4. Basic neural networks and optimization

- [ ] `Parameter`, base `Module`, and parameter registration and collection.
- [ ] `Sequential`, `Linear`, `ReLU`, `Sigmoid`, `Tanh`.
- [ ] MSE and numerically stable cross-entropy.
- [ ] SGD, momentum, and Adam.
- [ ] Training loop: forward → loss → backward → parameter update → gradient reset.
- [ ] `train()` and `eval()` modes, and the `Dropout` layer.

**Outcome:** linear regression, XOR, and a fully connected network for MNIST.

### 5. Data and model persistence

- [ ] Simple `Dataset` and `DataLoader`.
- [ ] Shuffling, batches, and handling the final incomplete batch.
- [ ] Reproducible experiments through random seed management.
- [ ] Save and load parameters using `state_dict()`.
- [ ] Save optimizer state to resume training.

**Outcome:** reproducible training with the ability to save and resume a model.

### 6. CNN

- [ ] `Conv2d`: a direct implementation followed by an `im2col` version.
- [ ] Padding, stride, and gradients for inputs, weights, and bias.
- [ ] `MaxPool2d`, `AvgPool2d`, and `Flatten`.
- [ ] `BatchNorm2d`, including training and evaluation behavior.
- [ ] A convolutional network for MNIST.

**Outcome:** an understanding of convolution and pooling, a working CNN, and a comparison with a fully connected network.

### 7. RNNs and sequence processing

- [ ] A simple `RNNCell` and hidden state.
- [ ] Process a sequence by unrolling the cell over time.
- [ ] Backpropagation through time (BPTT).
- [ ] Investigate vanishing and exploding gradients.
- [ ] Gradient clipping.
- [ ] Truncated BPTT with hidden state detached between segments.

**Outcome:** train a simple RNN on sequences and prepare for LSTM and GRU.

### 8. LSTM and GRU

- [ ] `LSTMCell`: gates, hidden state, and cell state.
- [ ] `GRUCell`: gates and hidden state updates.
- [ ] `LSTM` and `GRU` layers for sequence processing.
- [ ] Multiple recurrent layers.
- [ ] Character-level language models using RNN, LSTM, and GRU.

**Outcome:** understand memory mechanisms and compare recurrent architectures on the same task.

### 9. Transformer components

- [ ] `Embedding` and gradient accumulation for repeated tokens.
- [ ] Numerically stable `softmax`.
- [ ] `LayerNorm`, `RMSNorm`, `GELU`, and `SiLU`.
- [ ] Learned positional embeddings, then RoPE.
- [ ] Text data preparation and simple character-level tokenization.

**Outcome:** a set of operations and layers for building a Transformer.

### 10. Attention: MHA, MQA, and GQA

- [ ] Single-head scaled dot-product attention with `Q`, `K`, and `V` projections.
- [ ] Padding and causal masks.
- [ ] MHA (Multi-Head Attention): separate query, key, and value heads.
- [ ] MQA (Multi-Query Attention): one shared key/value pair for query heads.
- [ ] GQA (Grouped-Query Attention): shared key/value heads within groups of query heads.
- [ ] A common interface for attention variants and checks for intermediate tensor shapes.
- [ ] Attention map visualization.

**Outcome:** implementations of MHA, MQA, and GQA with verified gradients.

### 11. Decoder-only Transformer

- [ ] Transformer block: attention, residual connections, normalization, and feed-forward network.
- [ ] A standard MLP, then SwiGLU.
- [ ] Assemble a model from multiple blocks and an output projection to the vocabulary.
- [ ] Train for next-token prediction with correctly shifted inputs and targets.
- [ ] A small character-level language model.

**Outcome:** train a decoder-only Transformer on a text dataset.

### 12. Generation and inference optimization

- [ ] Greedy decoding, temperature, top-k, and top-p sampling.
- [ ] KV-cache for autoregressive generation.
- [ ] Check that logits match between cached generation and full-context recomputation.
- [ ] Compare MHA, MQA, and GQA: parameters, KV-cache size, and speed.
- [ ] Investigate the effects of positional representations and normalization.

**Outcome:** text generation and measurable architecture comparisons, accounting for implementation limitations.

## Checks and completion criteria

Each stage is complete when it includes:

- Implemented operations and required gradients.
- Checks on small examples with known results.
- Numerical gradient checks using finite differences.
- Comparisons of corresponding operations and layers with PyTorch.
- A working usage or training example.
- A brief explanation of the mathematics and implementation decisions.

Pay particular attention to tensor shapes, broadcasting, reused graph nodes, gradient accumulation, and numerical stability.

## End-to-end experiments

| Task | What it checks |
| --- | --- |
| Linear regression | Autograd and optimizers |
| XOR | Nonlinearities and multilayer networks |
| MNIST: MLP and CNN | Batches, classification, and convolutional layers |
| Next-character prediction: RNN, LSTM, GRU | Recurrent models and state handling |
| Next-character prediction: Transformer | Attention, positions, and generation |
| Transformer with MHA, MQA, and GQA | Attention variants and KV-cache cost |

To compare language architectures, use the same dataset, tokenization, and training/validation split; explicitly record differences in parameter count and training settings.

## Further exploration

- Blockwise attention and the ideas behind FlashAttention.
- Profiling and optimizing the most expensive operations.
- Gradually replacing NumPy operations with custom compute kernels.

Main trajectory: **scalar autograd → Tensor → basic neural networks → CNN → RNN → LSTM / GRU → Transformer → generation optimization**.
