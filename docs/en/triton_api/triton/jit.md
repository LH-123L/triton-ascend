# triton.jit

```python
triton.jit(fn: T) -> JITFunction[T]
triton.jit(*, version=None, repr: Callable | None = None, launch_metadata: Callable | None = None, do_not_specialize: Iterable[int] | None = None, debug: bool | None = None, noinline: bool | None = None) -> Callable[[T], JITFunction[T]]
```

A decorator that JIT-compiles a function using the Triton compiler.

- Note: When calling a JIT-compiled function, arguments with a `.data_ptr()` method and a `.dtype` attribute will be implicitly converted to pointers.

- Note: This function will be compiled and run on a GPU. It can only access the following:
    - Python primitives,
    - Built-in functions within the Triton package,
    - The arguments of this function,
    - Other JIT-compiled functions.

**Parameters:**`fn (Callable)` - The function to be JIT-compiled
