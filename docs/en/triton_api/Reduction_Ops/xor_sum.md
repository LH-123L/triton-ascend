# triton.language.xor_sum

## 1. OP Overview

Description: `triton.language.xor_sum` computes the XOR sum of the input tensor along the specified axis and returns the result of the XOR operation.

```python
triton.language.xor_sum(input, axis=None, keep_dims=False)
```

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter | Type | Description |
|--------|------|------|
| `input` | `Tensor` | Input tensor |
| `axis` | `int` or `None` | The dimension along which the XOR sum operation is performed. If None, performs the XOR operation over all dimensions |
| `keep_dims` | `bool` | If True, keeps the operated dimension with length 1 |

Return value:
`tensor`: The XOR sum of the input tensor along the specified axis, returning the result of the XOR operation

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
|---| ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
| Ascend A2/A3 | ✓ | ✓ | × | ✓ | × | ✓ | × | ✓ | × | × | × | ✓ |
| GPU support | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ |

#### 2.2.2 Shape Support

Conclusion: In terms of Shape, there is no difference between the GPU and Ascend platforms.

### 2.3 Special Limitations

> Capability missing relative to the community and cannot be implemented
> keep_dims=True needs to be tested with more specifications to determine whether it is fully supported. Currently, keep_dims=True is supported in the tested 3D dim=2 case.

### 2.4 Usage

The following example performs an xor_sum operation on a 2D-shape tensor:

```python
@triton.jit
def triton_xorsum_2d(in_ptr0, out_ptr0, dim: tl.constexpr, M: tl.constexpr, N: tl.constexpr, MNUMEL: tl.constexpr,
                     NNUMEL: tl.constexpr):
    mblk_idx = tl.arange(0, MNUMEL)
    nblk_idx = tl.arange(0, NNUMEL)
    mmask = mblk_idx < M
    nmask = nblk_idx < N
    mask = (mmask[:, None]) & (nmask[None, :])
    idx = mblk_idx[:, None] * N + nblk_idx[None, :]
    x = tl.load(in_ptr0 + idx, mask=mask, other=-float('inf'))
    tmp4 = tl.xor_sum(x, axis=dim)
    if dim == 0:
        tl.store(out_ptr0 + tl.arange(0, N), tmp4, None)
    else:
        tl.store(out_ptr0 + tl.arange(0, M), tmp4, None)

```
