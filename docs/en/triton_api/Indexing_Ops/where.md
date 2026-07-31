# triton.language.where

## 1. Function Overview

Description: Determines whether to return the value of tensor x or y based on the condition. When the condition is true, the value of x is returned; otherwise, the value of y is returned.

```python
triton.language.where(condition, x, y, _semantic=None)
```

## 2. Specifications

### 2.1 Parameter Description

| Parameter     | Type                | Description                                                             |
| ------------- | ------------------- | ---------------------------------------------------------------------- |
| `condition`   | `tensor(bool)`      | Tensor data                                                             |
| `x`           | `tensor`            | Tensor data                                                             |
| `y`           | `tensor`            | Tensor data                                                             |
| `_semantic`   | -                   | Reserved parameter, not supported for external calls                    |

Return value:
`out`: The output tensor has the same shape as the input x

### 2.2 OP Specifications

#### 2.2.1 DataType Support

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | √     | √      | √     | √      |  √      |  √       |  √       | √      | √    | √   | √    | √    | √    |
| Ascend A2/A3 | √    | √     | √     | √     | ×     | ×      | ×      | √     | √    | √    | ×    | √    | √    |

Conclusion: Ascend lacks support for uint and fp64 types compared to GPU.

#### 2.2.2 Shape Support

|        | Supported Dimension Range          |
| ------ | --------------- |
| GPU    | Only supports 1~5 dimensional tensors |
| Ascend A2/A3| Only supports 1~5 dimensional tensors |

Conclusion: In terms of Shape, there is no difference between the GPU and Ascend platforms; both support 1 to 5 dimensional tensors.

### 2.3 Special Limitations

> Capability missing relative to the community and cannot be implemented

Ascend lacks support for uint and fp64 types compared to GPU.

### 2.4 Usage

The following example performs element-wise selection based on the condition `X < Y`: when the condition is true, the element of `X` is taken; otherwise, the constant `1` is taken.

```python
@triton.jit
def fn_npu_(output_ptr, x_ptr, y_ptr, z_ptr,
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
    Y = tl.load(y_ptr + idx)

    tmp2 = X < Y
    ret = tl.where(tmp2, X, 1)

    tl.store(output_ptr + idx, ret)

x = test_common.generate_tensor(shape, dtype).npu()
y = test_common.generate_tensor(shape, dtype).npu()
```
