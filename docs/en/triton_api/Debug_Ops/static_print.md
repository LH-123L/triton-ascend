# triton.language.static_print

## 1. Function Overview

`static_print` is used to print information at compile time, similar to Python's `print()` function, but it executes during kernel compilation rather than at runtime.

```python
triton.language.static_print(*values, sep: str = ' ', end: str = '\n', file=None, flush=False, _semantic=None)
```

## 2. Specifications

### 2.1 Parameter Description

| Parameter | Type | Default Value | Description |
|------|------|--------|----------|
| `values`| `constexpr` | Required | The values to print, multiple parameters supported (must be compile-time constants) |
| `sep` | `str` | `' '` | The separator between values |
| `end` | `str` | `'\n'` | The suffix at the end of the print output |
|`file` | - | - | The file object to write to |
|`flush` | `bool` | `False` | Whether to flush the output buffer |
|`_semantic` | - | - | Reserved parameter, external invocation not supported for now |

### 2.2.1 Data Type Support

A3:

| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|------|-------|-------|-------|-------|--------|--------|--------|-------|------|------|------|------|------|
| GPU | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Ascend A2/A3 | ✓ | ✓ | ✓ | × | × | ×| × | ✓ | ✓ | ✓ | × | ✓ | ✓ |

### 2.2.2 Shape Support

|        | Supported Dimension Range          |
| ------ | --------------- |
| GPU    | Only 1~5-dimensional tensors supported |
| Ascend | Only 1~5-dimensional tensors supported |

Conclusion: In terms of Shape, there is no difference between the GPU and Ascend platforms; both support 1- to 5-dimensional tensors.

### 2.3 Special Limitations

> Capability missing relative to the community and cannot be implemented

Compared with GPU, Ascend lacks support for uint8, uint16, uint32, uint64, and fp64 (hardware limitation).

### 2.4 Usage

```python
import triton.language as tl

@triton.jit
def basic_static_print_example(x_ptr, BLOCK_SIZE: tl.constexpr):
    # Print the value of the constant at compile time
    tl.static_print("BLOCK_SIZE =", BLOCK_SIZE)
    tl.static_print(BLOCK_SIZE)
    # f-string printing is supported
    tl.static_print(f"BLOCK_SIZE={BLOCK_SIZE}")
```

If a **non-constant** result is printed, a value in the format `data type[data shape (empty for scalars)]` will be printed. For example, if the data type pointed to by `x_ptr` in the following code is `int32`, it will print `val:int32[constexpr[4]]`.

```python
import triton.language as tl

@triton.jit
def basic_static_print_example(x_ptr, BLOCK_SIZE: tl.constexpr):
    idx = tl.arange(0, 4)
    val = tl.load(x_ptr + idx)
    tl.static_print("val:",val)
    # f-string printing is not supported for non-constants
    #tl.static_print(f"val:{val}")
```
