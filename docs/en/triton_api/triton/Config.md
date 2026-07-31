# triton.Config

```python
class triton.Config(self, kwargs, num_warps, num_stages, num_ctas, maxnreg, pre_hook)
```

An object representing a kernel configuration that the autotuner may try

**Variables:**

- kwargs – A dictionary of meta-parameters to be passed to the kernel as keyword arguments.

- num_warps – The number of threads the kernel will use when compiled for a GPU. For example, if num_warps=8, each kernel instance will be automatically parallelized to execute cooperatively using 8 * 32 = 256 threads.

- num_stages – The number of stages the compiler should use when software-pipelining loops. Useful for matrix multiplication workloads on SM80+ GPUs.

- num_ctas - The number of blocks in a block cluster. Only applicable to SM90+.

- maxnreg - The maximum number of registers a single thread can use. Corresponds to the `.maxnreg` directive in PTX. Not supported on all platforms.

- pre_hook – A function that will be called before the kernel is invoked. The argument of this function is args.

```python
__init__(self, kwargs, num_warps=4, num_stages=2, num_ctas=1, maxnreg=None, pre_hook=None)
```

**Methods:**

| init(self, kwargs[, num_warps, ...]) |
|-----|
|all_kwargs (self)|
