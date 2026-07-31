# triton.language.flip

## 1. Function Overview

Description: Flips the tensor along a specified dimension.

```python
triton.language.flip(x, dim=None)
```

## 2. Specifications

### 2.1 Parameter Description

| Parameter     | Type                | Description                                                             |
| ------------- | ------------------- | ---------------------------------------------------------------------- |
| `x`           | `tensor`            | Tensor data                                                             |
| `dim`         | `int`               | Integer                                                                 |
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

|            | Supported Dimension Range        |
| ---------- | ------------------- |
| GPU        | Only supports 1~5 dimensional tensors |
| Ascend A2/A3 | Only supports 1~5 dimensional tensors |

Conclusion: In terms of Shape, there is no difference between the GPU and Ascend platforms; both support 1 to 5 dimensional tensors.

### 2.3 Special Limitations

> Capability missing relative to the community and cannot be implemented

Ascend lacks support for uint and fp64 types compared to GPU.

### 2.4 Usage

The following example flips the input tensor `x` along the specified dimension:

```python
@triton.jit
def fn_npu_3d(output_ptr, x_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, XNUMEL: tl.constexpr,
            YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    xidx = tl.arange(0, XB) + tl.program_id(0) * XB
    yidx = tl.arange(0, YB) + tl.program_id(1) * YB
    zidx = tl.arange(0, ZB) + tl.program_id(2) * ZB
    idx = xidx[:, None, None] * YNUMEL * ZNUMEL + yidx[None, :, None] * ZNUMEL + zidx[None, None, :]
    X = tl.load(x_ptr + idx)
    ret = tl.flip(X, 2)
    oidx = xidx[:, None, None] * YNUMEL * ZNUMEL + yidx[None, :, None] * ZNUMEL + zidx[None, None, :]
    tl.store(output_ptr + oidx, ret)

x = test_common.generate_tensor(shape, dtype).npu()
```
