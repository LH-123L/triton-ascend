# triton.language.reduce

## 1. OP Overview

Description: `triton.language.reduce` applies `combine_fn` to the input tensor along the specified axis `axis` for reduction and returns the reduced result tensor.

```python
triton.language.reduce(input, axis, combine_fn, keep_dims=False, _semantic=None, _generator=None)
```

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter | Type | Description |
|--------|------|------|
| `input` | `Tensor` or `tuple of Tensor` | Input tensor, which can be a single tensor or a tuple of tensors |
| `axis` | `int` or `None` | The dimension along which the reduce operation is performed. If None, reduces all dimensions |
| `combine_fn` | `Callable` | Function used to combine two groups of scalar tensors (must be decorated with @triton.jit) |
| `keep_dims` | `bool` | If True, keeps the reduced dimension with length 1 |
| `_semantic` | `Optional[str]` | Reserved parameter, not supported for external calls |
| `_generator` | `Optional[Generator]` | Reserved parameter, not supported for external calls |

**Note**: This function can also be called as a member function of a tensor, e.g., `x.reduce(...)` instead of `reduce(x, ...)`

Return value:
`tensor`: The result tensor after reduction along the specified axis `axis`.

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
|---| ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
| Ascend A2/A3 | ✓ | ✓ | × | ✓ | × | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓ |
| GPU support | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

#### 2.2.2 Shape Support

Conclusion: In terms of Shape, there is no difference between the GPU and Ascend platforms.

### 2.3 Special Limitations

> Capability missing relative to the community and cannot be implemented
> keep_dims=True needs to be tested with more specifications to determine whether it is fully supported. Currently, keep_dims=True is supported in the tested 3D dim=2 case.

### 2.4 Usage

The following example performs a reduce computation on a 2D-shape tensor, where combine_fn uses simple addition:

```python
@triton.jit
def _reduce_combine(a, b):
    return a + b

@triton.jit
def tt_reduce_2d(in_ptr, out_ptr,
                 xnumel: tl.constexpr, ynumel: tl.constexpr, znumel: tl.constexpr,
                 XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, dim: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    idx = xidx[:, None] * ynumel + yidx[None, :]

    x = tl.load(in_ptr + idx)
    ret = tl.reduce(x, dim, _reduce_combine)

    if dim == 0:
        oidx = yidx
    else:
        oidx = xidx
    tl.store(out_ptr + oidx, ret)

```
