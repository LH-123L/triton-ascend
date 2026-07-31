# triton.language.sum

## 1. OP Overview

Description: `triton.language.sum` computes the element-wise sum of the input tensor along the specified axis and returns the summation result.

```python
triton.language.sum(input, axis=None, keep_dims=False)
```

> **Version difference note**
>
> The `dtype` parameter is a feature introduced in community Triton 3.5.0. The currently released Triton-Ascend is developed based on community Triton 3.2.0 and does not have the `dtype` parameter. When upgraded to community Triton 3.5.0 in the future, the `dtype` parameter functionality will be fully supported.

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter | Type | Description |
|--------|------|------|
| `input` | `Tensor` | Input tensor |
| `axis` | `int` or `None` | The dimension along which the summation operation is performed. If None, sums over all dimensions |
| `keep_dims` | `bool` | If True, keeps the summed dimension with length 1 |

Return value:
`tensor`: Computes the element-wise sum of the input tensor along the specified axis and returns the summation result.

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
> The `dtype` parameter is not supported in the current version. In community Triton 3.5.0, the `dtype` parameter is used to control the accumulation data type of the summation operation: when not specified, integer types with bit width less than 32 are automatically promoted to `int32`/`uint32` to avoid overflow; when explicitly specified, the input is first converted to the specified type and then summed. The current Triton-Ascend is based on community Triton 3.2.0, and this type promotion logic is not yet supported; it will be fully supported when upgraded to 3.5.0 in the future.

### 2.4 Usage

The following example performs a sum operation on a 2D-shape tensor:

```python
@triton.jit
def tt_sum_2d(in_ptr, out_ptr,
              xnumel: tl.constexpr, ynumel: tl.constexpr, znumel: tl.constexpr,
              XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, dim: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    idx = xidx[:, None] * ynumel + yidx[None, :]

    x = tl.load(in_ptr + idx)
    ret = tl.sum(x, dim)

    if dim == 0:
        oidx = yidx
    else:
        oidx = xidx
    tl.store(out_ptr + oidx, ret)

```
