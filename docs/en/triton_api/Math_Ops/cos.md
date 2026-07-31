# triton.language.cos

## 1. Function Overview

Description: Computes the element-wise cosine of the input x.
Function Prototype (Triton 3.4.0 version):

```python
triton.language.cos(x, _semantic=None)
```

## 2. Specifications

### 2.1 Parameter Description

| Parameter           | Type                | Description                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `x`        | `tensor`          | Tensor data                                                      |
| `_semantic`   | -                 | Reserved parameter; not yet supported for external calls

Return Value:
`x`: The output tensor has the same shape as the input tensor x

### 2.2 OP Specifications

#### 2.2.1 DataType Support

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | ×    | ×     | ×     | ×     | ×     | ×      | ×      | ×     | ×    | ×    | √    | √    | ×    |
| Ascend A2/A3 | ×    | ×     | ×     | ×     | ×     | ×      | ×      | ×     | √    | √    | ×    | √    | ×    |

Conclusion: Compared with GPU, Ascend lacks fp64 support but additionally supports fp16 and fp32.
torch_npu supports uint8.

#### 2.2.2 Shape Support

|        | Supported Dimension Range          |
| ------ | --------------- |
| GPU    | Only 1~5-dimensional tensors supported |
| Ascend A2/A3 | Only 1~5-dimensional tensors supported |

Conclusion: There is no difference between the GPU and Ascend platforms in terms of Shape; both support 1 to 5-dimensional tensors.

### 2.3 Special Restrictions

> Features missing relative to the community that cannot be implemented

None.

### 2.4 Usage Example

The following example computes the element-wise cosine of the input tensor `x`:

```python
@triton.jit
def fn_npu_(output_ptr, x_ptr,
            XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr,
            XNUMEL: tl.constexpr, YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    zoffs = tl.program_id(2) * ZB

    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    zidx = tl.arange(0, ZB) + zoffs

    idx = xidx[:, None, None] * YNUMEL * ZNUMEL + yidx[None, :, None] * ZNUMEL + zidx[None, None, :]

    X = tl.load(x_ptr + idx)

    ret = tl.cos(X)

    tl.store(output_ptr + idx, ret)
```
