# triton.language.tensor.__xor__

## 1. Function Overview

Description: Computes the XOR value of two elements.

```python
# Via the operator
x ^ y

# Or directly call the dunder method
x.__xor__(y)
```

## 2. Specifications

### 2.1 Parameter Description

| Parameter           | Type                | Description                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `x`        | `tensor`          | Tensor data                                                      |
| `y`        | `tensor`          | Tensor data                                                      |
| `_semantic`   | -                 | Reserved parameter, external invocation not supported for now

Return value:
`out`: A tensor with the same shape as x and y

### 2.2 OP Specifications

#### 2.2.1 DataType Support

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | √     | √      | √     | √      |  ×      |  ×       |  ×       | √      | ×    | ×    | ×    | ×    | √    |
| Ascend A2/A3 | √    | √     | √     | √     | ×     | ×      | ×      | √     | ×    | ×    | ×    | ×    | √    |

Conclusion: Compared with GPU, Ascend lacks uint type support.

#### 2.2.2 Shape Support

|        | Supported Dimension Range          |
| ------ | --------------- |
| GPU    | Only 1~5-dimensional tensors supported |
| Ascend A2/A3 | Only 1~5-dimensional tensors supported |

Conclusion: In terms of Shape, there is no difference between the GPU and Ascend platforms; both support 1- to 5-dimensional tensors.

### 2.3 Special Limitations

> Capability not yet supported relative to the community

Compared with GPU, Ascend lacks uint type support.

### 2.4 Usage

The following example computes the element-wise XOR of two input tensors:

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

    ret = X ^ Y

    tl.store(output_ptr + idx, ret)

x = test_common.generate_tensor(shape, dtype).npu()
y = test_common.generate_tensor(shape, dtype).npu()
```
