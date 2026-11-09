# Educational examples

The [scalar_autograd.ipynb](scalar_autograd.ipynb) notebook illustrates the computation graph, backward pass, and gradient accumulation for scalar `Value` objects. To run it from the project root:

```bash
uv sync --group notebook
uv run --group notebook jupyter lab
```

Scripts will be added as framework components become available:

| Planned script | Task |
| --- | --- |
| `scalar_autograd.py` | Derivatives of a scalar expression |
| `linear_regression.py` | Train a linear regression model |
| `xor.py` | A nonlinear multilayer network |
| `mnist_mlp.py` | A fully connected network on MNIST |
| `mnist_cnn.py` | A convolutional network on MNIST |
| `char_rnn.py` | A character-level model using RNN, LSTM, or GRU |
| `char_transformer.py` | A decoder-only Transformer using MHA, MQA, or GQA |
| `generate.py` | Generate text from a saved model |

Complete task-specific models and training loops belong in examples; reusable layers belong in `tinytorch.nn`.

Each example should support a configurable seed, have explicit run parameters, and include brief instructions. Store datasets in the root `data/` directory, weights in `checkpoints/`, and experiment results in `runs/`. These directories are excluded from Git. Data downloads must be initiated explicitly, with no import-time side effects.
