# triton.language.swizzle2d

## 1. Function Overview

Description: **Converts the indices of a row-major matrix of size size_i × size_j into column-major matrix indices, in groups of size_g rows each.**

```python
triton.language.swizzle2d(i, j, size_i, size_j, size_g)
```

## 2. Specifications

### 2.1 Parameter Description

| Parameter     | Type                | Description                                                             |
| ------------- | ------------------- | ---------------------------------------------------------------------- |
| `i`           | `tensor`            | Index value; the maximum value is size_i-1                              |
| `j`           | `tensor`            | Index value; the maximum value is size_j-1                              |
| `size_i`      | `int`               | Integer, indicating the length of the index value i                     |
| `size_j`      | `int`               | Integer, indicating the length of the index value j                     |
| `size_g`      | `int`               | Integer                                                                 |

Return value:
`out0, out1`: Tensors with the same shape as i and j

### 2.2 OP Specifications

#### 2.2.1 DataType Support

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | ×     | ×      | √     | ×      |  ×      |  ×       |  ×       | √      | ×    | ×   | ×    | ×    | ×    |
| Ascend A2/A3 | ×    | ×     | √     | ×     | ×     | ×      | ×      | √     | ×    | ×   | ×    | ×   | ×

#### 2.2.2 Shape Support

|        | Supported Dimension Range          |
| ------ | --------------- |
| GPU    | Only supports 2-dimensional tensors |
| Ascend A2/A3 | Only supports 2-dimensional tensors |

Conclusion: In terms of Shape, there is no difference between the GPU and Ascend platforms; both support 2-dimensional tensors.

### 2.3 Special Limitations

> Capability missing relative to the community and cannot be implemented

None.

### 2.4 Usage

The following example converts the indices of a row-major matrix into column-major matrix indices, in groups of `size_g` rows each:

```python
@triton.jit
def fn_npu_(out0, out1, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    i = tl.arange(0, XB)[:, None]
    j = tl.arange(0, YB)[None, :]
    ij = i * YB + j
    xx, yy = tl.swizzle2d(i, j, size_i=XB, size_j=YB, size_g=ZB)

    ptr = tl.load(out0)
    xx = tl.cast(xx, dtype=ptr.dtype)
    yy = tl.cast(yy, dtype=ptr.dtype)
    tl.store(out0 + ij, xx)
    tl.store(out1 + ij, yy)
```
