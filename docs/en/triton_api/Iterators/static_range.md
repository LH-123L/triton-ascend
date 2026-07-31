# triton.language.static_range

## 1. Function Overview

`static_range` is a static-range iterator, similar to `range`, but it aggressively performs loop unrolling optimization at compile time.

```python
triton.language.static_range(arg1, arg2=None, step=None, _semantic=None)
```

## 2. Specifications

### 2.1 Parameter Description

| Parameter | Type | Default Value | Description |
|------|------|--------|----------|
| `arg1` | `constexpr` | Required | The starting value (when a single argument is given, it serves as the ending value, starting from 0) |
| `arg2` | `constexpr` | - | The ending value (not included in the range) |
| `step` | `constexpr` | `1` | The step increment for each iteration |
| `_semantic` | - | - | Reserved parameter, external invocation not supported for now |

### 2.2 Type Support

A3:

| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|------|-------|-------|-------|-------|--------|--------|--------|-------|------|------|------|------|------|
| GPU | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | × | × |
| Ascend 910 series | ✓ | ✓ | ✓ | ×|×| × | × | ✓ | × | × | × | × | × |

### 2.3 Special Limitations

> Capability missing relative to the community and cannot be implemented

Compared with GPU, Ascend lacks support for uint8, uint16, uint32, uint64, and fp64 (hardware limitation).

### 2.4 Usage

```python
@triton.jit
def optimized_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    # Use static_range for small-scale loop unrolling to eliminate loop overhead
    for i in tl.static_range(BLOCK_SIZE):
        # When BLOCK_SIZE is a compile-time constant, the entire loop will be unrolled
        x = tl.load(x_ptr + i)
        y = x * x
        tl.store(y_ptr + i, y)

    # For comparison: using range incurs loop control overhead
    for i in tl.range(BLOCK_SIZE):
        # This loop has loop control logic at runtime
        x = tl.load(x_ptr + i)
        y = x * x
        tl.store(y_ptr + i, y)
```

`static_range` trades code size for runtime performance and is suitable for scenarios where the loop count is known and small.
