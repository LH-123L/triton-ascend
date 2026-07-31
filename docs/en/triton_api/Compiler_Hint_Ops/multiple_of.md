# triton.language.multiple_of

## 1. Function Overview

`multiple_of` is used to declare to the compiler that the first value in the input tensor is a multiple of a certain number.

```python
triton.language.multiple_of(input, values, _semantic=None)
```

## 2. Specifications

### 2.1 Parameter Description

| Parameter | Type | Default Value | Description |
|------|------|--------|----------|
| `input` | `Tensor` | Required | Input tensor whose values are all multiples of certain numbers |
| `values` | `constexpr[int]` or `list[constexpr[int]]` | Required | Declares that the input values are multiples of these numbers (a single integer or a sequence of integers per dimension) |
| `_semantic` | - | - | Reserved parameter, external invocation not supported for now |

**`values` describes the divisibility characteristics of the first value in each dimension, so the dimensionality of `values` must be the same as that of `input`.
Note the dimensionality reduction that occurs when the last dimension of `shape` is `1`.**

For example, a two-dimensional `input` corresponds to a generic `values` argument of `[1,1]`.

### 2.2 Type Support

A3:

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
def basic_multiple_of_example(A, B, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    input_data = tl.load(A + offsets)

    # Declare that the first value of the input tensor is a multiple of BLOCK_SIZE
    input_data = tl.multiple_of(input_data, [BLOCK_SIZE])
```
