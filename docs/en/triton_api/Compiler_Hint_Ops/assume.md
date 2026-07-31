# triton.language.assume

## 1. Function Overview

`assume` is used to provide conditional assumption information to the compiler, allowing the compiler to perform optimizations based on conditions known to be true. This is a compiler hint operation and does not check the condition at runtime.

```python
triton.language.assume(cond, _semantic=None)
```

## 2. Specifications

### 2.1 Parameter Description

| Parameter | Type | Default Value | Description |
|------|------|--------|----------|
| `cond` | `bool` | Required | The condition expression that the compiler may assume to be true |
| `_semantic` | - | - | Reserved parameter, external invocation not supported for now |

### 2.2 Type Support

A3:

| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|------|-------|-------|-------|-------|--------|--------|--------|-------|------|------|------|------|------|
| GPU | × | × | × | × | × | × | × | × | × | × | × | × | ✓ |
| Ascend A2/A3 | × | × | × | × | × | × | × | × | × | × | × | × | ✓ |

### 2.3 Usage

The `assume` operation allows developers to help the compiler generate more efficient code while ensuring correctness.

```python
import triton.language as tl

@triton.jit
def basic_assume_example(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    # Assume BLOCK_SIZE is a power of 2, so the compiler can optimize division based on this
    tl.assume((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0)

    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)

    # The compiler knows BLOCK_SIZE is a power of 2 and can optimize division into a shift operation
    result = x // BLOCK_SIZE + y % BLOCK_SIZE
    tl.store(y_ptr + offsets, result)
```
