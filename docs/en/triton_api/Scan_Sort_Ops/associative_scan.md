# triton.language.associative_scan

## 1. OP Overview

Description: `triton.language.associative_scan` applies an associative scan operation to the input tensor along the specified axis, using the combine_fn function to combine elements and update the carry value.

```python
triton.language.associative_scan(input, axis, combine_fn, reverse=False, _semantic=None, _generator=None)
```

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter | Type | Description |
|--------|------|------|
| `input` | `Tensor` or `tuple of Tensor` | Input tensor, which can be a single tensor or a tuple of tensors |
| `axis` | `int` | The dimension along which the associative scan operation is performed |
| `combine_fn` | `Callable` | Function used to combine two groups of scalar tensors (must be decorated with @triton.jit) |
| `reverse` | `bool` | Whether to apply the associative scan in the reverse direction along the axis |
| `_semantic` | `Optional[str]` | Reserved parameter, not supported for external calls |
| `_generator` | `Optional[Generator]` | Reserved parameter, not supported for external calls |

Return value:
`tensor`: The tensor after applying the associative scan operation to the input tensor along the specified axis, combining elements with the combine_fn function and updating the carry value.

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
|---| ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
| GPU support | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Ascend A2/A3 | ✓ | ✓ | × | ✓ | × | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓ |

#### 2.2.2 Shape Support

Conclusion: In terms of Shape, there is no difference between the GPU and Ascend platforms.

### 2.3 Special Limitations

> Capability missing relative to the community and cannot be implemented
> reverse=True applies the associative scan in the reverse direction along the axis. This feature requires the data loaded by tl.load to be aligned, i.e., no mask is used to filter out redundant data indices, as shown in the following example code:

```python
    tl.static_assert(
        numel_x == XBLOCK, "numel_x must be equal to XBLOCK in this kernel"
    )
    tl.static_assert(
        numel_r == RBLOCK, "numel_r must be equal to RBLOCK in this kernel"
    )
    idx_x = tl.arange(0, XBLOCK)
    idx_r = tl.arange(0, RBLOCK)
    idx = idx_x[:, None] * numel_r + idx_r[None, :]
    x = tl.load(in_ptr0 + idx)
```

### 2.4 Usage

The following example performs an associative_scan operation on a 2D-shape tensor:

```python

@triton.jit
def bitwise_and_fn(a, b):
    return a & b


@triton.jit
def bitwise_or_fn(a, b):
    return a | b


@triton.jit
def bitwise_xor_fn(a, b):
    return a ^ b


@triton.jit
def minimum_fn(a, b):
    return tl.minimum(a, b)


@triton.jit
def maximum_fn(a, b):
    return tl.maximum(a, b)

@triton.jit
def triton_kernel_2d_scan(
        out_ptr0,
        in_ptr0,
        dim: tl.constexpr,
        reverse: tl.constexpr,
        numel_x: tl.constexpr,
        numel_r: tl.constexpr,
        XBLOCK: tl.constexpr,
        RBLOCK: tl.constexpr,
        combine_fn_name: tl.constexpr,
):
    tl.static_assert(
        numel_x == XBLOCK, "numel_x must be equal to XBLOCK in this kernel"
    )
    tl.static_assert(
        numel_r == RBLOCK, "numel_r must be equal to RBLOCK in this kernel"
    )
    idx_x = tl.arange(0, XBLOCK)
    idx_r = tl.arange(0, RBLOCK)
    idx = idx_x[:, None] * numel_r + idx_r[None, :]
    x = tl.load(in_ptr0 + idx)

    if combine_fn_name == "maximum_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=maximum_fn)
    elif combine_fn_name == "minimum_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=minimum_fn)
    elif combine_fn_name == "bitwise_or_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=bitwise_or_fn)
    elif combine_fn_name == "bitwise_xor_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=bitwise_xor_fn)
    elif combine_fn_name == "bitwise_and_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=bitwise_and_fn)
    tl.store(out_ptr0 + idx, ret)

```
