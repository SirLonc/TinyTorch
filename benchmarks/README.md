# Performance benchmarks

Add benchmarks after the corresponding implementations have passed correctness checks.

Planned comparisons:

- Direct convolution and the `im2col` version.
- RNN, LSTM, and GRU with identical input shapes.
- MHA, MQA, and GQA: parameter count, KV-cache size, and generation time.
- Generation with KV-cache and full-context recomputation.

Record the seed, shapes, dtype, dependency versions, hardware, and number of warmups and repetitions. Save results in `runs/`. Run benchmarks separately from correctness tests.
