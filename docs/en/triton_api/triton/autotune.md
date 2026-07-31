# triton.autotune

```python
triton.autotune(configs, key, prune_configs_by=None, reset_to_zero=None, restore_value=None, pre_hook=None, post_hook=None, warmup=25, rep=100, use_cuda_graph=False)
```

A decorator for automatically tuning a triton.jit function.

```python
@triton.autotune(configs=[
    triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4),
    triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8),
  ],
  key=['x_size']  # Whenever the value of x_size changes, the two configurations above are evaluated.
)
@triton.jit
def kernel(x_ptr, x_size, **META):
    BLOCK_SIZE = META['BLOCK_SIZE']
```

- Note: When all configurations are evaluated, the kernel will be run multiple times. That is, any values updated by the kernel will be updated multiple times. To avoid this unwanted behavior, you can use the `reset_to_zero` parameter, which resets the values of the provided tensors to zero before running any configuration.
- Note: If the environment variable `TRITON_PRINT_AUTOTUNING` is set to `"1"`, Triton will print a message to standard output (stdout) after each autotuned kernel, including the time spent on autotuning and the best configuration.

**Parameters:**

- `configs (list[triton.Config])` - A list of `triton.Config` objects.
- `key (list[str])` - A list of argument names whose changed values will trigger evaluation of all configurations.
- `prune_configs_by (dict)` - A dictionary of functions to prune configurations. It contains the following fields:
  - `'perf_model'`: The performance model used to predict the running time of different configurations; returns the running time
  - `'top_k'`: The number of configurations to benchmark
  - `'early_config_prune'` (optional): A function used to prune configurations early (e.g., `num_stages`). It takes `configs: List[Config]` as input and returns the pruned configurations
- `reset_to_zero (list[str])` - A list of parameter names that will be reset to zero before any configuration is evaluated.
- `restore_value (list[str])` - A list of parameter names whose values will be restored after any configuration is evaluated.
- `pre_hook (lambda args, reset_only)` - A function that will be called before the kernel is invoked. This parameter overrides the default `pre_hook` of `reset_to_zero` and `restore_value`.
  - `args`: The list of arguments passed to the kernel
  - `reset_only`: A boolean indicating whether `pre_hook` is only used to reset values, without a corresponding `post_hook`
- `post_hook (lambda args, exception)` - A function that will be called after the kernel is invoked. This parameter overrides the default `post_hook` of `restore_value`.
  - `args`: The list of arguments passed to the kernel
  - `exception`: The exception raised by the kernel in case of a compilation or runtime error
- `warmup (int)` - The warmup time (in milliseconds) passed to the benchmark, with a default value of 25.
- `rep (int)` - The repetition time (in milliseconds) passed to the benchmark, with a default value of 100.
- `use_cuda_graph (bool)` - Whether to use CUDA Graph for performance measurement (defaults to `False`).
