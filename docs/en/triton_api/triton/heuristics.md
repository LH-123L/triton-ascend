# triton.heuristics

```python
triton.heuristics(values)
```

A decorator for specifying how certain meta-parameter values are computed. This is useful in cases where autotuning is too expensive or not applicable.

```python
@triton.heuristics(values={'BLOCK_SIZE': lambda args: 2 ** int(math.ceil(math.log2(args[1])))})
@triton.jit
def kernel(x_ptr, x_size, **META):
    BLOCK_SIZE = META['BLOCK_SIZE'] # smallest power-of-two >= x_size
```

**Parameters:** `values (dict[str, Callable[[list[Any]], Any]]**)` - A dictionary containing meta-parameter names and functions that compute the meta-parameter values. Each such function takes a list of positional arguments as input.
