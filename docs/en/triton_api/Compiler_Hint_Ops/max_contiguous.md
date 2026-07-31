# triton.language.max_contiguous

## 1. Function Overview

`max_contiguous` is used to declare the contiguity pattern of the input tensor to the compiler, informing the compiler that the first `value` elements of the input tensor are contiguous.

```python
triton.language.max_contiguous(input, values, _builder=None, _semantic=None)
```

## 2. Specifications

### 2.1 Parameter Description

| Parameter | Type | Default Value | Description |
|------|------|--------|----------|
| `input` | `Tensor` | Required | Input tensor whose memory accesses have a specific contiguity pattern |
| `values` | `constexpr[int]` or `list[constexpr[int]]` | Required | Compile-time constant integer (or sequence of integers) describing the contiguity pattern |
| `_semantic` | - | - | Reserved parameter, external invocation not supported for now |

**`values` describes the contiguity characteristics of each dimension, so the dimensionality of `values` must be the same as that of `input`.
Note the dimensionality reduction that occurs when the last dimension of `shape` is `1`.**

For example, a two-dimensional `input` corresponds to a generic `values` argument of `[1,1]`.

### 2.2 Type Support

| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|------|-------|-------|-------|-------|--------|--------|--------|-------|------|------|------|------|------|
| GPU | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Ascend A2/A3 | ✓ | ✓ | ✓ | × | × | ×| × | ✓ | ✓ | ✓ | × | ✓ | ✓ |

### 2.3 Special Limitations

> Capability missing relative to the community and cannot be implemented

Compared with GPU, Ascend lacks support for uint8, uint16, uint32, uint64, and fp64 (hardware limitation).

### 2.4 Usage

```python
@triton.jit
def triton_max_contiguous(A, B, BLOCK_SIZE : tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    val = tl.load(A + offsets)
    # Declare that the first BLOCK_SIZE elements in offset are contiguous
    input_data = tl.max_contiguous(val, [BLOCK_SIZE])

    # The compiler can generate more efficient memory access instructions
    result = input_data* 2
    tl.store(B + offsets, result)
```
