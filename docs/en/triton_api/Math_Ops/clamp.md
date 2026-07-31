# triton.language.clamp

## 1. Function Overview

Description: Clamps the range of the tensor x to [min, max].

```python
triton.language.clamp(x, min, max, propagate_nan: constexpr = PropagateNan.NONE, _semantic=None)
```

## 2. Specifications

### 2.1 Parameter Description

| Parameter           | Type                | Description                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `x`        | `tensor`          | Tensor data                                                      |
| `min`       | `tensor`       | Lower bound (can be a tensor or scalar, broadcast to the shape of `x`) |
| `max`       | `tensor`       | Upper bound (can be a tensor or scalar, broadcast to the shape of `x`) |
| `propagate_nan` | `triton.language.core.constexpr` | Whether to propagate NaN for min or max                                              |
| `_semantic`   | -                 | Reserved parameter; not yet supported for external calls

Return Value:
`x`: The output tensor has the same shape as the input tensor x

### 2.2 OP Specifications

#### 2.2.1 DataType Support

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | ×    | ×     | ×     | ×     | ×     | ×      | ×      | ×     | √    | √    | √    | √    | ×    |
| Ascend A2/A3 | ×    | ×     | ×     | ×     | ×     | ×      | ×      | ×     | √    | √    | ×    | √    | ×    |

#### 2.2.2 Shape Support

|        | Supported Dimension Range          |
| ------ | --------------- |
| GPU    | Only 1~5-dimensional tensors supported |
| Ascend | Only 1~5-dimensional tensors supported |

Conclusion: There is no difference between the GPU and Ascend platforms in terms of Shape; both support 1 to 5-dimensional tensors.

### 2.3 Special Restrictions

> Features missing relative to the community that cannot be implemented

Ascend lacks fp64 support compared with GPU.

#### 2.3.1 propagate_nan Parameter Restrictions

**Note: When `propagate_nan=tl.PropagateNAN.NONE`, the system automatically adds NaN value handling logic, which results in:**

1. **Increased UB space usage**: The additional NaN detection and handling requires more UB space
2. **Possible performance degradation**: The additional computation logic may degrade operator execution performance

**Suggestion:**

- If the input data contains no NaN values, or strict NaN handling semantics are not required, it is recommended to use the default value or select an appropriate `propagate_nan` parameter value based on actual requirements
- In scenarios where UB space is limited, special attention should be paid to this parameter to avoid compilation failure due to insufficient UB space

### 2.4 Usage Example

The following example performs the clamp operation on the input tensor `x`:

```python
@triton.jit
def tt_clamp_2d(in_ptr, out_ptr, min_ptr, max_ptr,
                   xnumel: tl.constexpr, ynumel: tl.constexpr, znumel: tl.constexpr,
                   XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
       xoffs = tl.program_id(0) * XB
       yoffs = tl.program_id(1) * YB
       xidx = tl.arange(0, XB) + xoffs
       yidx = tl.arange(0, YB) + yoffs
       idx = xidx[:, None] * ynumel + yidx[None, :]

       x = tl.load(in_ptr + idx)
       min_ = tl.load(min_ptr + idx)
       max_ = tl.load(max_ptr + idx)
       ret = tl.clamp(x, min_, max_)

       tl.store(out_ptr + idx, ret)
```
