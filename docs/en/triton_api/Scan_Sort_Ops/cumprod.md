# triton.language.cumprod

## 1. OP Overview

Description: `triton.language.cumprod` computes the cumulative product of the input tensor along the specified axis and returns the cumulative product result.

```python
triton.language.cumprod(input, axis=0, reverse=False)
```

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter | Type | Description |
|--------|------|------|
| `input` | `Tensor` | Input tensor |
| `axis` | `int` | The dimension along which the cumulative product operation is performed; defaults to 0 |
| `reverse` | `bool` | If True, performs the cumulative product operation in the reverse direction |

The `cumprod` function computes the cumulative product (prefix product) along the specified axis. For example, for the input `[a, b, c, d]`, the cumulative product result is `[a, a*b, a*b*c, a*b*c*d]`.

When `reverse=True`, the reverse cumulative product is computed: `[a*b*c*d, b*c*d, c*d, d]`.

Unlike `cumsum`, `cumprod` does not have a `dtype` parameter, so attention should be paid to data type overflow when using it, especially for cumulative products of integer types.

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
|---| ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
| Ascend A2/A3 | ✓ | ✓ | × | ✓ | × | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓ |
| GPU support | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

#### 2.2.2 Shape Support

Conclusion: In terms of Shape, there is no difference between the GPU and Ascend platforms.

### 2.3 Special Limitations

### 2.4 Usage

The following example performs a cumprod operation on a 2D-shape tensor:

```python
@triton.jit
def triton_kernel_2d(
        out_ptr0,
        in_ptr0,
        dim: tl.constexpr,
        reverse: tl.constexpr,
        numel_x: tl.constexpr,
        numel_r: tl.constexpr,
        XBLOCK: tl.constexpr,
        RBLOCK: tl.constexpr,
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
    ret = tl.cumprod(x, axis=dim, reverse=reverse)
    tl.store(out_ptr0 + idx, ret)
```
