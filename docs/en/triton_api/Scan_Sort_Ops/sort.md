# triton.language.sort

## 1. Function Overview

Description: Sorts the input tensor `x` along a dimension in ascending or descending order.

```python
triton.language.sort(x, dim: constexpr | None = None, descending: constexpr = False)
```

## 2. Specifications

### 2.1 Parameter Description

| Parameter     | Type                | Description                                                             |
| ------------- | ------------------- | ---------------------------------------------------------------------- |
| `x`           | `tensor`            | Tensor data                                                             |
| `dim`         | `int`               | The dimension to sort along                                              |
| `descending`  | `bool`              | Whether to sort in descending order                                      |

Return value:
`x`: The output tensor has the same shape as the input x

### 2.2 OP Specifications

#### 2.2.1 DataType Support

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | √    | √     | √      | √     | ×     | ×      | ×      | √     | √    | √    | √    | √    | √    |
| Ascend A2/A3 | √     | √      | ×     | ×     | ×     | ×      | ×      | ×     | √    | √    | ×    | √    | ×    |

Conclusion: Ascend lacks support for int32, uint8, int64, float64, and bool compared to GPU.
torch_npu supports uint8.

#### 2.2.2 Shape Support

|        | Supported Dimension Range          |
| ------ | --------------- |
| GPU    | Only supports 1~5 dimensional tensors |
| Ascend A2/A3 | Only supports 1~5 dimensional tensors |

Conclusion: In terms of Shape, there is no difference between the GPU and Ascend platforms; both support 1 to 5 dimensional tensors.

### 2.3 Special Limitations

> Capability missing relative to the community and cannot be implemented

Due to Bisheng compiler limitations, int32, uint8, int64, float64, and bool cannot be implemented.

### 2.4 Usage

The following example sorts the input tensor `x`:

```python
@triton.jit
def sort_kernel_2d(X, Z, N: tl.constexpr, M: tl.constexpr, descending: tl.constexpr):
    pid = tl.program_id(0)
    offx = tl.arange(0, M)
    offy = pid * M
    off2d = offx + offy
    x = tl.load(X + off2d)
    x = tl.sort(x, dim=0, descending=descending)
    tl.store(Z + off2d, x)
```
