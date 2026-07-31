# triton.language.cumsum

## 1. OP Overview

Description: `triton.language.cumsum` computes the cumulative sum of the input tensor along the specified axis and returns the cumulative summation result.

```python
triton.language.cumsum(input, axis=0, reverse=False)
```

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter | Type | Description |
|--------|------|------|
| `input` | `Tensor` | Input tensor |
| `axis` | `int` | The dimension along which the cumulative sum operation is performed; defaults to 0 |
| `reverse` | `bool` | If True, performs the cumulative sum operation in the reverse direction |

The `cumsum` function computes the cumulative sum (prefix sum) along the specified axis. For example, for the input `[a, b, c, d]`, the cumulative sum result is `[a, a+b, a+b+c, a+b+c+d]`.

When `reverse=True`, the reverse cumulative sum is computed: `[a+b+c+d, b+c+d, c+d, d]`.

Return value:
`tensor`: Computes the cumulative sum of the input tensor along the specified axis and returns the cumulative summation result.

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

The following example performs a cumsum operation on a 2D-shape tensor:

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
    ret = tl.cumsum(x, axis=dim, reverse=reverse)
    tl.store(out_ptr0 + idx, ret)
```
